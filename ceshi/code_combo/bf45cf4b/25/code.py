# Combo variant of bf45cf4b (index 25)
# combo_spec: D1/0+D6/0+D15/0
# Name: 反转空洞奇偶镜像间隔铺贴
#
# Design analysis
# ===============
# a) Concrete behavior and trace:
#    The input inventory stays the same as the original task: one monochrome
#    control sprite with bg holes and one multicolor square tile, disconnected
#    under 8-connectivity. The fused output rule changes three stages at once.
#    I make the parity rule concrete as a checkerboard based on `(r+c) % 2`:
#      - filled ctrl slot: paste the tile; odd parity slots use `fliplr(tile)`
#      - hole ctrl slot: paste the inverted tile; odd parity slots use
#        `fliplr(inverted(tile))`
#      - adjacent logical slots are separated by a 1-pixel bg line
#    Example with tile T = [[1,2],[3,4]], inv(T) = [[8,7],[6,5]],
#    ctrl mask = [[x,bg],[bg,x]]:
#      original bf45cf4b:
#        [[T][..]
#         [..][T]]
#      fused variant:
#        [[T].[flip(inv(T))]
#         [.............]
#         [inv(T)].[flip(T)]]
#    The differences are therefore visible in all three axes:
#      slot content for holes changes from empty -> inverted tile,
#      odd parity slots are mirrored,
#      and every tile row/column is spaced apart by 1 bg pixel.
#
# b) Complexity calibration:
#    I keep only two input objects, so object recognition remains stable. To
#    keep the fusion observable rather than noisy, the generator enforces:
#      - tile uses only 2-3 colors from disjoint inversion pairs
#      - tile is not horizontally symmetric, so the parity mirror is visible
#      - ctrl has both fills and holes on both parity classes, so all four slot
#        types (filled-even, filled-odd, hole-even, hole-odd) appear
#      - gap is always exactly 1, not sampled, so the spatial rule is crisp
#
# c) Hidden-assumption audit of the original main():
#    - `find_components(..., monochromatic=False) == 2` still holds because the
#      input object inventory is unchanged and placement remains separated.
#    - Sorting by color-count and squareness still identifies tile vs control;
#      the generator keeps control monochrome and tile multicolor square.
#    - `ctrl_cols[0]` still assumes exactly one control color; preserved.
#    - The original output allocation `ch*bh by cw*bw` is invalid now because
#      D15 inserts separators, so the correct shape is
#      `ch*bh + (ch-1)` by `cw*bw + (cw-1)`.
#    - The original "only paste on filled slots" assumption is invalid because
#      D1 requires hole slots to receive an inverted tile.
#    - The original "every pasted block is identical" assumption is invalid
#      because D6 requires parity-dependent horizontal reflection.
#    These three old assumptions all break in the same inner loop, so the
#    fused solver uses one shared slot-selection function.
#
# d) Placement legality:
#    The generator still builds the output first, then places the tile object
#    and control sprite into a 12-18 by 12-18 input canvas with at least a
#    1-pixel bg gap and no overlap. Control is pasted transparently so its
#    internal holes remain bg. Because the tile is fully foreground, it stays a
#    single connected multicolor component.
#
# e) Visual observability:
#    A human can compare repeated copies of one tile and see:
#      - the same geometry appears in normal and mirrored orientations
#      - hole locations are not blank; they carry the same geometry in a stable
#        complementary palette `c -> 9-c`
#      - the assembled mosaic is segmented by thin bg seams
#    Requiring both parity classes for fills and holes means the examples show
#    each branch of the fused rule directly, rather than leaving some branch
#    implicit.
#
# f) Visual cleanliness:
#    The output has exactly one layer of symbols per logical slot plus uniform
#    bg separators. There are no extra markers, borders, or overlaps. Avoiding
#    bg inside the tile prevents confusion between true separator lines and tile
#    interior holes.
#
# g) Change inferability:
#    The color rule is restricted to the fixed global map `c -> 9-c`, and tile
#    colors are sampled from only one side of each inversion pair. This makes
#    normal and inverted palettes disjoint, so the learner can infer that hole
#    slots are transformed versions of the same tile rather than unrelated
#    independent patterns.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


INV_PAIRS = ((0, 9), (1, 8), (2, 7), (3, 6), (4, 5))


def _invert_tile(tile: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(tile, dtype=int, copy=True)
    mask = (out != int(bg))
    out[mask] = 9 - out[mask]
    return out


def _slot_categories(ctrl: np.ndarray, bg: int) -> dict[str, int]:
    counts = {"filled_even": 0, "filled_odd": 0, "hole_even": 0, "hole_odd": 0}
    h, w = map(int, ctrl.shape)
    for r in range(h):
        for c in range(w):
            filled = int(ctrl[r, c]) != int(bg)
            parity = "even" if (r + c) % 2 == 0 else "odd"
            key = f"{'filled' if filled else 'hole'}_{parity}"
            counts[key] += 1
    return counts


def _assemble_variant(
    tile: np.ndarray,
    ctrl: np.ndarray,
    bg: int,
    *,
    gap: int,
    mirror: bool,
    invert_holes: bool,
) -> np.ndarray:
    bh, bw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out_h = int(ch * bh + (ch - 1) * gap)
    out_w = int(cw * bw + (cw - 1) * gap)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    inv_tile = _invert_tile(tile, bg)

    for r in range(ch):
        for c in range(cw):
            filled = int(ctrl[r, c]) != int(bg)
            if filled:
                patch = tile
            elif invert_holes:
                patch = inv_tile
            else:
                continue

            if mirror and (r + c) % 2 == 1:
                patch = np.fliplr(patch)

            paste(out, patch, r * (bh + gap), c * (bw + gap))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a monochrome control sprite into a spaced tile mosaic.

    Fused rule:
    - ctrl filled slot -> tile, mirrored on odd checkerboard parity
    - ctrl hole slot -> inverted tile, mirrored on odd checkerboard parity
    - neighboring logical slots are separated by 1 bg pixel
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

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)

    return _assemble_variant(block, ctrl, bg, gap=1, mirror=True, invert_holes=True)


def _pick_tile_colors(bg: int, ctrl_color: int, rng: np.random.Generator) -> list[int]:
    usable_pairs = [
        pair
        for pair in INV_PAIRS
        if int(bg) not in pair and int(ctrl_color) not in pair
    ]
    n_tile_colors = int(rng.integers(2, min(3, len(usable_pairs)) + 1))
    chosen = rng.choice(len(usable_pairs), size=n_tile_colors, replace=False)
    colors = []
    for idx in chosen:
        a, b = usable_pairs[int(idx)]
        colors.append(int(a if rng.integers(0, 2) == 0 else b))
    return colors


def _make_tile(bg: int, ctrl_color: int, rng: np.random.Generator) -> np.ndarray:
    tile_colors = _pick_tile_colors(bg, ctrl_color, rng)
    n_colors = len(tile_colors)

    for _ in range(100):
        size = int(rng.integers(3, 6))
        indices = rng.integers(0, n_colors, size=(size, size))

        for i in range(n_colors):
            rr, cc = divmod(i, size)
            indices[rr, cc] = i

        if np.array_equal(indices, np.fliplr(indices)):
            rr = int(rng.integers(0, size))
            cc = int(rng.integers(0, size - 1))
            left = int(indices[rr, cc])
            right = int(indices[rr, size - 1 - cc])
            if left == right:
                indices[rr, size - 1 - cc] = int((right + 1) % n_colors)

        tile = np.vectorize(lambda x: tile_colors[int(x)], otypes=[int])(indices).astype(int)
        inv_tile = _invert_tile(tile, bg)
        if np.array_equal(tile, np.fliplr(tile)):
            continue
        if len(colors_present(tile, exclude={bg})) != n_colors:
            continue
        if np.any(tile == int(bg)):
            continue
        if np.any(inv_tile == int(bg)) or np.any(inv_tile == int(ctrl_color)):
            continue
        if set(colors_present(tile, exclude={bg})) & set(colors_present(inv_tile, exclude={bg})):
            continue
        return tile

    raise RuntimeError("failed to build observable asymmetric tile")


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


def _fill_component_count(mask: np.ndarray) -> int:
    comps = find_components(mask.astype(int), bg=0, connectivity=8, monochromatic=False, return_format="dict")
    return len(comps)


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        area = h * w
        min_fill = max(3, (area + 1) // 2)
        max_fill = min(area - 2, int(np.floor(0.72 * area)))
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
            pick_r, pick_c = frontier[int(rng.integers(len(frontier)))]
            mask[pick_r, pick_c] = True

        if int(mask.sum()) != target:
            continue
        if _fill_component_count(mask) != 1:
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)
        cats = _slot_categories(ctrl, bg)
        if min(cats.values()) <= 0:
            continue
        return ctrl

    raise RuntimeError("failed to build nondegenerate control sprite")


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    return _assemble_variant(tile, ctrl, bg, gap=1, mirror=True, invert_holes=True)


def _compose_no_mirror(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    return _assemble_variant(tile, ctrl, bg, gap=1, mirror=False, invert_holes=True)


def _compose_no_gap(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    return _assemble_variant(tile, ctrl, bg, gap=0, mirror=True, invert_holes=True)


def _compose_no_inversion(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    return _assemble_variant(tile, ctrl, bg, gap=1, mirror=True, invert_holes=False)


def _compose_original(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * bh, c * bw)
    return out


def _variant_metric(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> dict[str, int]:
    cats = _slot_categories(ctrl, bg)
    combo = _compose_output(tile, ctrl, bg)
    no_mirror = _compose_no_mirror(tile, ctrl, bg)
    no_gap = _compose_no_gap(tile, ctrl, bg)
    no_inv = _compose_no_inversion(tile, ctrl, bg)
    orig = _compose_original(tile, ctrl, bg)
    return {
        "mirror_pixels": int(np.count_nonzero(tile != np.fliplr(tile))),
        "filled_even": int(cats["filled_even"]),
        "filled_odd": int(cats["filled_odd"]),
        "hole_even": int(cats["hole_even"]),
        "hole_odd": int(cats["hole_odd"]),
        "gap_delta_area": int(combo.size - no_gap.size),
        "diff_vs_no_mirror": int(not np.array_equal(combo, no_mirror)),
        "diff_vs_no_inversion": int(not np.array_equal(combo, no_inv)),
        "diff_vs_original": int(not np.array_equal(combo, orig)),
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
    bg = int(rng.integers(0, 10))
    ctrl_candidates = [c for c in range(10) if c != bg]
    ctrl_color = int(rng.choice(ctrl_candidates))

    tile = _make_tile(bg, ctrl_color, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
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

    metric = _variant_metric(tile, ctrl, bg)
    if metric["mirror_pixels"] <= 0:
        return None
    if min(metric["filled_even"], metric["filled_odd"], metric["hole_even"], metric["hole_odd"]) <= 0:
        return None
    if metric["gap_delta_area"] <= 0:
        return None
    if metric["diff_vs_no_mirror"] != 1 or metric["diff_vs_no_inversion"] != 1 or metric["diff_vs_original"] != 1:
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
