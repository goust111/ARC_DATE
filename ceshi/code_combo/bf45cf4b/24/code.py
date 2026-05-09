# Combo variant of bf45cf4b (index 24)
# combo_spec: D5/2+D15/0
# Name: 间隔排列居中固定网格
#
# Design analysis
# - a) Concrete behavior and trace:
#   Baseline bf45cf4b identifies two disconnected input objects: one multicolor
#   square tile and one monochrome control sprite with bg holes. The original
#   output stamps the tile into a tight grid: for a 2x2 tile T and control mask
#   [[1,0,1],[1,1,0]], the baseline output is 4x6.
#   In this combo, assembly changes in two coordinated stages:
#   1. D15/0: keep the same logical mask, but insert a 1-pixel bg separator
#      between every adjacent logical tile row and column. The same example
#      becomes 5x8 because rows use starts 0 and 3, cols use starts 0, 3, 6.
#   2. D5/2: do not return that tight spaced rectangle directly. Instead place
#      it centered inside a fixed 20x20 bg canvas. If the spaced result is 5x8,
#      its top-left anchor is ((20-5)//2, (20-8)//2) = (7, 6).
#   The visible rule is therefore: "expand with regular bg grid lines, then
#   center the whole enlarged sprite inside a constant outer frame."
# - b) Complexity calibration:
#   The combo is intentionally limited to two changes only: internal spacing and
#   outer fixed-canvas centering. There are no extra markers, no color remaps,
#   and no multiple canvas sizes. To keep the change visually obvious, the
#   generator enforces control sizes >= 3 in both axes and forces the spaced
#   assembly to be at most 18x18, guaranteeing both separator lines and a
#   non-zero outer margin inside the 20x20 frame.
# - c) Hidden-assumption audit of the original main():
#   `find_components(...)=2` still holds because the input inventory is
#   unchanged. Sorting by color count and squareness still identifies tile vs
#   control because the tile remains multicolor and square while the control is
#   monochrome. `ctrl_cols[0]` still assumes control monochromaticity; generator
#   preserves that invariant and keeps ctrl color disjoint from tile colors.
#   The original layout assumptions that break are:
#     `out = np.full((ch * bh, cw * bw), ...)`
#     `paste(out, block, r * bh, c * bw)`
#   They assume zero inter-tile gap and no outer canvas. In the combo they must
#   become:
#     1. assemble on a spaced lattice with step `(tile_size + 1)`;
#     2. center that assembled sprite in a fixed 20x20 canvas.
#   A new combo-specific assumption appears: the spaced assembly must fit inside
#   20x20. The generator guarantees that.
# - d) Placement legality:
#   Input placement is still independent of output layout. The generator places
#   the tile and control on a 12-18 by 12-18 bg canvas with at least a 1-pixel
#   bg gap, no overlap, and full in-bounds placement. Output centering never
#   feeds back into input placement, so there is no new collision mode there.
# - e) Visual observability:
#   Across a few examples, a human sees the same repeated multicolor tile, but
#   copies no longer touch because 1-pixel bg seams split the lattice. On top of
#   that, every output is a 20x20 square with the constructed mosaic floating in
#   the middle on bg. Since the generator enforces positive outer margins and
#   control dimensions >= 3, neither the spacing nor the centering can collapse
#   back to the original task visually.
# - f) Visual cleanliness:
#   The output has only two layers: the centered spaced sprite and the bg frame.
#   There is no overlap between symbols, and the separator lines use the same bg
#   color already established by the task, so the picture stays regular rather
#   than cluttered.
# - g) Change inferability:
#   No color mapping is introduced. All reasoning comes from shape/layout:
#   repeated tile copies, uniform 1-pixel seams, and a constant 20x20 outer
#   frame. This keeps the rule inferable from 2-4 ARC examples without relying
#   on color ids or hidden ordering tricks.
# - Post-edit assumption audit:
#   `_assemble_with_gaps()` assumes the control bbox is the logical lattice and
#   every non-bg control cell is a filled slot; generator preserves rectangular
#   ctrl bboxes with holes. `_center_on_fixed_canvas()` assumes the assembled
#   sprite height and width are each <= 20; generator chooses tile/control sizes
#   from that inequality directly. The component-ranking logic assumes the tile
#   has more fg colors than the control and remains square; generator validates
#   that before accepting a sample.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


OUTPUT_SIZE = 20
GAP = 1


def _assemble_with_gaps(block: np.ndarray, ctrl: np.ndarray, bg: int, gap: int = GAP) -> np.ndarray:
    """Expand filled ctrl cells into `block` copies with uniform bg separators."""
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    out_h = int(ch * bh + (ch - 1) * gap)
    out_w = int(cw * bw + (cw - 1) * gap)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, block, r * (bh + gap), c * (bw + gap))
    return out


def _center_on_fixed_canvas(sprite: np.ndarray, bg: int, size: int = OUTPUT_SIZE) -> np.ndarray:
    """Center `sprite` inside a fixed square bg canvas."""
    sh, sw = map(int, sprite.shape)
    if sh > size or sw > size:
        raise ValueError(f"assembled sprite {sprite.shape} does not fit in {size}x{size} canvas")
    out = np.full((size, size), int(bg), dtype=int)
    r0 = (size - sh) // 2
    c0 = (size - sw) // 2
    paste(out, sprite, r0, c0)
    return out


def _tight_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    spaced = _assemble_with_gaps(tile, ctrl, bg, gap=GAP)
    return _center_on_fixed_canvas(spaced, bg, size=OUTPUT_SIZE)


def _tight_shape(tile: np.ndarray, ctrl: np.ndarray) -> tuple[int, int]:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    return int(ch * th), int(cw * tw)


def _spaced_shape(tile: np.ndarray, ctrl: np.ndarray, gap: int = GAP) -> tuple[int, int]:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    return int(ch * th + (ch - 1) * gap), int(cw * tw + (cw - 1) * gap)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Identify the multicolor square block and the monochrome control sprite.
    2. Expand each filled control pixel into one full tile copy.
    3. Insert 1-pixel bg separators between adjacent logical tile rows/cols.
    4. Center the assembled spaced sprite inside a fixed 20x20 bg canvas.
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
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((OUTPUT_SIZE, OUTPUT_SIZE), int(bg), dtype=int)

    ctrl_col = int(ctrl_cols[0])
    mask_like_ctrl = np.where(ctrl == int(ctrl_col), int(ctrl_col), int(bg))
    return _compose_output(block, mask_like_ctrl, bg)


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


def _make_tile(bg: int, fg_colors: list[int], size: int, rng: np.random.Generator) -> np.ndarray:
    n_tile_colors = int(rng.integers(2, min(4, len(fg_colors)) + 1))
    tile_colors = list(rng.choice(fg_colors, size=n_tile_colors, replace=False))
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.full((size, size), int(bg), dtype=int)
    assignments: dict[int, int] = {}
    for i, color in enumerate(tile_colors):
        assignments[i] = int(color)

    interior_idxs = [i for i, (_cells, border) in enumerate(orbits) if not border and i not in assignments]
    if interior_idxs and rng.random() < 0.25:
        assignments[int(rng.choice(interior_idxs))] = int(bg)

    for i, (_cells, border) in enumerate(orbits):
        if i not in assignments:
            palette = tile_colors if border else tile_colors + ([bg] if interior_idxs else [])
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


def _make_ctrl(bg: int, color: int, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    area = h * w
    min_fill = max((area + 1) // 2, 1)
    max_fill = min(area - 1, int(np.floor(0.72 * area)))
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


def _choose_sizes(rng: np.random.Generator) -> tuple[int, int, int]:
    tile_size = int(rng.integers(2, 6))
    # Need `ctrl_dim * tile_size + (ctrl_dim - 1) <= OUTPUT_SIZE - 2`
    # so the spaced assembly fits inside 20x20 with at least one bg margin.
    max_ctrl_dim = (OUTPUT_SIZE - 1) // (tile_size + GAP)
    if max_ctrl_dim < 3:
        raise ValueError("tile size leaves no valid control dimension")
    ctrl_h = int(rng.integers(3, max_ctrl_dim + 1))
    ctrl_w = int(rng.integers(3, max_ctrl_dim + 1))
    return tile_size, ctrl_h, ctrl_w


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


def _variant_metric(tile: np.ndarray, ctrl: np.ndarray) -> dict[str, int]:
    tight_h, tight_w = _tight_shape(tile, ctrl)
    sh, sw = _spaced_shape(tile, ctrl, gap=GAP)
    top = (OUTPUT_SIZE - sh) // 2
    left = (OUTPUT_SIZE - sw) // 2
    bottom = OUTPUT_SIZE - sh - top
    right = OUTPUT_SIZE - sw - left
    return {
        "tight_area": int(tight_h * tight_w),
        "spaced_area": int(sh * sw),
        "spacing_delta_area": int(sh * sw - tight_h * tight_w),
        "separator_lines": int((ctrl.shape[0] - 1) + (ctrl.shape[1] - 1)),
        "canvas_padding_area": int(OUTPUT_SIZE * OUTPUT_SIZE - sh * sw),
        "min_outer_margin": int(min(top, bottom, left, right)),
        "assembled_h": int(sh),
        "assembled_w": int(sw),
        "output_constant_size": int(OUTPUT_SIZE),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = list(rng.choice(np.arange(10, dtype=int), size=5, replace=False))
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_palette = [int(x) for x in palette[2:]]

    tile_size, ctrl_h, ctrl_w = _choose_sizes(rng)
    tile = _make_tile(bg, tile_palette, tile_size, rng)
    ctrl = _make_ctrl(bg, ctrl_color, ctrl_h, ctrl_w, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)

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

    metric = _variant_metric(tile, ctrl)
    if metric["separator_lines"] <= 0:
        return None
    if metric["spacing_delta_area"] <= 0:
        return None
    if metric["canvas_padding_area"] <= 0 or metric["min_outer_margin"] < 1:
        return None
    if metric["assembled_h"] > 18 or metric["assembled_w"] > 18:
        return None

    if expected.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
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
