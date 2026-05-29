# Variant of bf45cf4b
# Dimension: D1 — 像素放大方式 (transform)
# Change: 控制精灵中每个填充像素替换为完整的多色方块瓦片副本，每个空洞像素替换为等大的背景区域... → 根据控制精灵中填充像素的邻域数量选择不同瓦片变体
#
# Design analysis
# a) Concrete behavior:
#    Input still contains exactly two disconnected foreground components: one
#    monochrome control sprite with bg holes inside its bbox, and one square
#    multicolor tile. The change happens only at expansion time. Each filled
#    control pixel is expanded to a transformed copy of the tile based on the
#    number of filled 4-neighbors that control pixel has inside the control
#    bbox. Mapping used here:
#      0 -> original tile
#      1 -> rotate 90 deg CCW
#      2 -> rotate 180 deg
#      3 -> rotate 270 deg CCW
#      4 -> left-right flip
#    Small example: if a control pixel is an endpoint of a 4-neighbor path,
#    old logic pasted the raw tile; new logic pastes the 90-degree rotated
#    tile. If another pixel is fully surrounded on four sides, old logic again
#    pasted the raw tile; new logic uses the mirrored tile instead.
# b) Complexity calibration:
#    A symmetric tile would make rotations/flips invisible, so the generator
#    must produce an asymmetric square tile whose five mapped variants are all
#    distinct. The control sprite must also exhibit several neighbor-count
#    categories, otherwise the output would visually collapse toward the
#    original task. The generator therefore requires at least three distinct
#    neighbor counts, with repeated count-1 and count-2 examples and at least
#    one extreme count (0 or 4).
# c) Old implicit assumptions that break:
#    - `paste(out, block, ...)` assumed every filled control cell used the same
#      tile. This no longer holds; the pasted patch depends on local 4-neighbor
#      count, so `main()` must compute per-cell counts before pasting.
#    - The old generator used dihedral-symmetric tiles, which were harmless for
#      fixed copying but would make the new transforms invisible. The tile
#      generator must therefore forbid those symmetries.
#    - The old `_compose_output()` ignored control adjacency. The variant
#      output builder must match solver logic exactly or output-first
#      verification becomes vacuous.
# d) Placement legality:
#    Input placement still uses two disconnected components with at least one
#    full bg row/column gap, so 8-connectivity does not merge them. The tile
#    keeps a fully foreground border so its observed bbox remains square even if
#    it contains an interior bg hole. The control mask is cropped to its true
#    bbox before placement so `main()` sees exactly the intended rectangle.
# e) Visual observability:
#    Human solvers can compare repeated endpoint / corridor / junction tiles
#    across examples and see that the tile orientation changes with local
#    degree, not with global position. Because the tile is asymmetric, the
#    rotations and mirror are directly visible in 2-4 examples.
# f) Visual cleanliness:
#    No extra markers are added. The scene remains the original clean layout:
#    one control sprite and one tile in a mostly empty grid. The variation is
#    confined to the expanded output, which stays tightly cropped and easy to
#    scan.
# g) Color inference:
#    There is no new color remapping. Colors only move through geometric
#    transforms, keeping the deduction burden on shape/orientation rather than
#    arbitrary palette changes.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


def _neighbor_count4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    total = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            total += 1
    return total


def _tile_variant(tile: np.ndarray, count4: int) -> np.ndarray:
    if count4 == 0:
        return tile
    if count4 == 1:
        return np.rot90(tile, 1)
    if count4 == 2:
        return np.rot90(tile, 2)
    if count4 == 3:
        return np.rot90(tile, 3)
    if count4 == 4:
        return np.fliplr(tile)
    raise ValueError(f"unexpected 4-neighbor count: {count4}")


def _split_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
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


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, *, use_variants: bool) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return out
    mask = (ctrl == int(ctrl_cols[0]))

    for r in range(ch):
        for c in range(cw):
            if not bool(mask[r, c]):
                continue
            patch = _tile_variant(tile, _neighbor_count4(mask, r, c)) if use_variants else tile
            paste(out, patch, r * th, c * tw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a monochrome control sprite by stamping transformed copies of a
    multicolor square tile. The chosen transform depends on the control pixel's
    filled 4-neighbor count within the control bbox.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    block, ctrl = _split_objects(g, bg)
    return _compose_output(block, ctrl, bg, use_variants=True)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _count_hist(mask: np.ndarray) -> dict[int, int]:
    hist: dict[int, int] = {}
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            k = _neighbor_count4(mask, r, c)
            hist[k] = hist.get(k, 0) + 1
    return hist


def _is_single_component(mask: np.ndarray) -> bool:
    grid = mask.astype(int)
    comps = find_components(grid, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    return len(comps) == 1


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    for _ in range(400):
        size = int(rng.integers(4, 6))
        n_colors = int(rng.integers(3, min(4, len(fg_colors)) + 1))
        tile_colors = [int(x) for x in rng.choice(fg_colors, size=n_colors, replace=False)]
        tile = rng.choice(tile_colors, size=(size, size))

        if size >= 4 and rng.random() < 0.7:
            rr = int(rng.integers(1, size - 1))
            cc = int(rng.integers(1, size - 1))
            tile[rr, cc] = int(bg)

        if np.any(tile[0, :] == bg) or np.any(tile[-1, :] == bg) or np.any(tile[:, 0] == bg) or np.any(tile[:, -1] == bg):
            continue
        if len(colors_present(tile, exclude={bg})) < 3:
            continue
        if not _is_single_component(tile != bg):
            continue

        variants = [_tile_variant(tile, k) for k in range(5)]
        if len({v.tobytes() for v in variants}) != 5:
            continue
        return tile.astype(int)
    raise RuntimeError("failed to build asymmetric tile")


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(1200):
        h = int(rng.integers(5, 8))
        w = int(rng.integers(5, 8))
        fill_p = float(rng.uniform(0.40, 0.60))
        mask = rng.random((h, w)) < fill_p
        if not mask.any():
            continue
        if not _is_single_component(mask):
            continue

        mask = _crop_mask(mask)
        mh, mw = mask.shape
        area = mh * mw
        fill = int(mask.sum())
        fill_ratio = fill / area
        if not (3 <= mh <= 7 and 3 <= mw <= 7):
            continue
        if not (0.35 <= fill_ratio <= 0.72):
            continue
        if fill == area:
            continue

        hist = _count_hist(mask)
        if len(hist) < 3:
            continue
        if hist.get(1, 0) < 2 or hist.get(2, 0) < 2:
            continue
        if hist.get(0, 0) + hist.get(4, 0) < 1:
            continue

        ctrl = np.full((mh, mw), int(bg), dtype=int)
        ctrl[mask] = int(color)
        return ctrl
    raise RuntimeError("failed to build control sprite")


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
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    block_colors = palette[1:5]
    ctrl_color = int(palette[5])

    tile = _make_tile(bg, block_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    expected = _compose_output(tile, ctrl, bg, use_variants=True)
    uniform = _compose_output(tile, ctrl, bg, use_variants=False)
    if np.array_equal(expected, uniform):
        return None

    inp = _make_input(tile, ctrl, bg, rng)
    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))
    if sorted(len(cols) for cols in colors_per_comp) != [1, len(colors_present(tile, exclude={bg}))]:
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
