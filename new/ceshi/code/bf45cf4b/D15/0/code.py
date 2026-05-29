# Variant of bf45cf4b
# Dimension: D15 — 输出组装的空间布局 (grid_shape)
# Change: 瓦片紧密网格排列，无间隙，形成连续的放大图案... → 瓦片之间添加1像素背景色间隔
#
# Design analysis
# - Reference baseline:
#   I read `/home/lixinyu/snap/new_code/solvers/arc2/58f5dbd5.py` before editing.
#   Its relevant mechanism is not object recognition but the packing layout:
#   tiles are placed on a larger background canvas with `step = tile_size + 1`,
#   which creates 1-pixel background gaps. For this variant I keep bf45cf4b's
#   own recognition rule and borrow only that spacing idea.
# - a) Concrete behavior and trace:
#   Input stays the same: exactly two disconnected foreground components, one
#   multicolor square tile and one monochrome rectangular control sprite with bg
#   holes. The original output packed logical tile slots tightly. Example with a
#   2x2 tile `T` and control mask `[[1,1],[1,0]]`:
#     original: [[T][T]
#                [T][bg-tile]]   -> size 4x4
#     variant : [[T].[T]
#                .....
#                [T].[bg-tile]]  -> size 5x5
#   where `.` denotes background grid lines. The gap is inserted between every
#   adjacent logical tile row/column of the control bbox, not only between two
#   filled tiles; this keeps the output as a regular assembled grid and makes
#   D15 a pure layout change.
# - b) Complexity calibration:
#   I do not add outer borders, new markers, or extra objects. Only internal
#   separators change. To make the change observable, the generator keeps
#   control sprites at least 3x3 with >50% fill, so neighboring logical cells
#   are common and the inserted grid lines are visible across repeated tiles.
# - c) Hidden-assumption audit of the original main():
#   `find_components(...)=2` still holds, because the input object inventory is
#   unchanged. Sorting by color count and squareness still identifies tile vs
#   control. `ctrl_cols[0]` still assumes the control is monochrome; generator
#   keeps that invariant. The lines that depended on the old tight layout were:
#     `out = np.full((ch * bh, cw * bw), ...)`
#     `paste(out, block, r * bh, c * bw)`
#   Those assume zero gap. They become wrong in the variant because they would
#   overwrite intended separator lines and produce the old shape. They are the
#   only solver lines that need semantic change.
# - d) Placement legality:
#   Generator placement of the two input objects is unchanged except that it now
#   validates against the spaced-output oracle. Objects are still placed with at
#   least a 1-pixel background gap in the input, with no overlap and within a
#   12-18 by 12-18 canvas.
# - e) Visual observability:
#   Across 2-4 examples, a human sees the same tile repeated according to the
#   control mask, but neighboring copies no longer touch; thin bg seams split
#   the mosaic. Because control sizes are >=3 in both axes, the spaced output is
#   always strictly larger than the old tight output, so the layout change
#   cannot silently degenerate back to the original task.
# - f) Visual cleanliness:
#   Only one layer is shown in the output: repeated copies of the tile plus bg
#   separators. There is no overlap between symbols. The bg seams are uniform
#   width 1, so the result stays regular rather than cluttered.
# - g) Color inferability:
#   No color rule changes. The tile's colors are copied verbatim; the only new
#   visual fact is the 1-pixel background spacing.
# - Variant metric for non-degeneration:
#   For control size `(ch, cw)` and tile size `(bh, bw)`, the layout expansion
#   over the original task is
#     delta_area = (ch * bh + ch - 1) * (cw * bw + cw - 1) - (ch * bh) * (cw * bw)
#   Since generator uses `ch, cw >= 3`, `delta_area > 0` always. I also measure
#   the count of inserted separator lines, `(ch - 1) + (cw - 1)`, when auditing
#   generated samples after implementation.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


def _assemble_with_gaps(block: np.ndarray, ctrl: np.ndarray, bg: int, gap: int = 1) -> np.ndarray:
    """Expand each filled ctrl cell to `block`, leaving 1px bg separators between logical slots."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale up a monochrome control sprite by stamping a polychrome tile at each filled pixel.

    Variant rule:
    - The logical tile grid is no longer tight.
    - Between every adjacent tile row/column, insert a 1-pixel-wide background separator.
    - Hole cells in the control bbox still correspond to empty tile-sized regions.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert not (len(comps) != 2)

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

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)

    ctrl_col = int(ctrl_cols[0])
    mask_like_ctrl = np.where(ctrl == int(ctrl_col), int(ctrl_col), int(bg))
    return _assemble_with_gaps(block, mask_like_ctrl, bg, gap=1)


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


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    n_tile_colors = int(rng.integers(2, 4))
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
    return _assemble_with_gaps(tile, ctrl, bg, gap=1)


def _tight_output_shape(tile: np.ndarray, ctrl: np.ndarray) -> tuple[int, int]:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    return int(ch * th), int(cw * tw)


def _variant_metric(tile: np.ndarray, ctrl: np.ndarray) -> dict[str, int]:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    tight_h, tight_w = _tight_output_shape(tile, ctrl)
    spaced_h = int(ch * th + (ch - 1))
    spaced_w = int(cw * tw + (cw - 1))
    return {
        "separator_lines": int((ch - 1) + (cw - 1)),
        "delta_area": int(spaced_h * spaced_w - tight_h * tight_w),
    }


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
    palette = list(rng.choice(np.arange(10, dtype=int), size=5, replace=False))
    bg = int(palette[0])
    fg = [int(x) for x in palette[1:]]

    tile = _make_tile(bg, fg, rng)
    ctrl = _make_ctrl(bg, fg[-1], rng)
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
    if metric["separator_lines"] <= 0 or metric["delta_area"] <= 0:
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
