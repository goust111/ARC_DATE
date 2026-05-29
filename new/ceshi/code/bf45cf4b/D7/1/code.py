# Variant of bf45cf4b
# Dimension: D7 — 空洞区域的填充方式 (transform)
# Change: 控制精灵中的空洞位置填充瓦片大小的纯背景色区域... → 空洞位置填充瓦片颜色的互补/映射版本

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


# Design analysis for the D7 variant
# ==================================
# Context:
# - The allowed reference files contain the original bf45cf4b solver/generator and ARC helpers.
# - The variant description cites source task 4c416de3 for "color_transfer", but within the allowed
#   files there is no readable solver source for 4c416de3, only its evaluation JSON. So the design
#   below borrows only the general mechanism "same geometry, recolored copy", not any hidden code.
#
# a) Concrete behavior trace
# - Original behavior:
#   control bbox cell = filled  -> stamp the original multicolor tile.
#   control bbox cell = hole/bg -> leave an all-bg tile-sized blank region.
# - Variant behavior:
#   control bbox cell = filled  -> stamp the original tile.
#   control bbox cell = hole/bg -> stamp a recolored version of the same tile.
# - To keep the recoloring visually recoverable from 2-4 ARC examples, the generator uses a tile with
#   exactly two non-bg colors A and B. The "negative" tile is formed by swapping A <-> B everywhere.
#   Geometry is unchanged; only the color assignment flips.
# - Small trace:
#   tile = [[A,B],[B,A]], ctrl bbox = [X,bg]
#   original output row: [tile][blank]
#   variant output row : [tile][swap(tile)] = [[A,B][B,A]][[B,A][A,B]]
#   The affected pipeline stage is only the output assembly at hole positions; object detection and
#   control-mask extraction stay the same.
#
# b) Complexity calibration
# - A fully arbitrary many-color mapping would be hard to infer from only a few examples.
# - Restricting the tile to exactly two non-bg colors keeps the rule sharp: hole tiles are obvious
#   color inversions, not unrelated new sprites.
# - We still keep nontriviality by requiring both filled and hole cells in the control sprite, so a
#   single output contains repeated evidence of both the original tile and the inverted tile.
#
# c) Implicit-assumption audit of main()
# - `find_components(..., monochromatic=False)` still assumes exactly two disconnected foreground
#   objects. The variant does not change this and the generator preserves it.
# - Sorting components by "more colors / more square" still works, but now the block must have exactly
#   two non-bg colors rather than >=2 arbitrary colors. The generator enforces that.
# - `mask = (ctrl == ctrl_col)` still assumes the control object is monochrome except for bg holes.
#   That remains true.
# - The original nested loop assumed the only meaningful branch was "filled => paste tile".
#   That was harmless before because holes meant leaving the prefilled bg untouched. Under D7, that
#   would silently fail because holes now need active stamping. So the loop must explicitly handle the
#   `else` branch and paste the swapped tile there.
#
# d) Placement legality
# - Input still contains exactly two 8-connected foreground components separated by at least one bg
#   cell. The tile has no internal bg, so it is a single compact component. The control sprite is a
#   connected monochrome rectangle-with-holes placed separately from the tile.
# - Output assembly is tight-cropped and deterministic because every control bbox cell expands to one
#   tile-sized region.
#
# e) Visual observability
# - A human sees two region types in the output: original tiles at filled control cells and color-
#   swapped tiles at hole cells. Because the same tile geometry repeats, the difference is attributable
#   to hole-vs-filled status rather than to a second unrelated sprite.
# - The generator requires at least two hole cells whenever possible, so the recolored branch is not a
#   single isolated accident.
#
# f) Visual cleanliness
# - No overlay markers or extra objects are introduced.
# - The tile itself has no bg holes in this variant, so the swapped-hole regions look like complete
#   "negative" tiles rather than mixed blank/color patches, which keeps the output easy to parse.
#
# g) Color-change inferability
# - Only two tile colors are involved, and the mapping is repeated many times in one sample:
#   A -> B and B -> A. This is enough structure for ARC-style induction.
#
# h) Generator anti-degeneration target
# - Quantitative metric: `variant_diff = number of output cells where variant output differs from the
#   original task's blank-hole output`.
# - For a valid non-degenerate sample we want:
#   1. hole_count >= 1
#   2. both tile colors occur
#   3. variant_diff > 0
#   The generator checks these conditions indirectly by construction, and the post-change test script
#   below reports their distribution explicitly.
#
# i) New main() assumptions and generator guarantees
# - Assumption: the block has exactly two non-bg colors.
#   Guarantee: `_make_tile` always uses exactly two distinct foreground colors and fills the whole tile.
# - Assumption: the control bbox contains at least one hole and at least one filled cell.
#   Guarantee: `_make_ctrl` rejects degenerate masks.


def _swap_tile_colors(tile: np.ndarray, bg: int) -> np.ndarray:
    cols = colors_present(tile, exclude={bg})
    assert len(cols) == 2
    a, b = map(int, cols)
    swapped = np.array(tile, copy=True)
    swapped[tile == a] = b
    swapped[tile == b] = a
    return swapped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale a monochrome control sprite by stamping the original tile at filled cells and the
    color-swapped tile at hole cells.
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
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)

    ctrl_col = int(ctrl_cols[0])
    mask = ctrl == ctrl_col
    inv_block = _swap_tile_colors(block, bg)

    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            tile = block if bool(mask[r, c]) else inv_block
            paste(out, tile, r * bh, c * bw)
    return out


def _dihedral_orbits(size: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
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
            out.append(cells)
    return out


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    del bg  # Tile is intentionally fully filled in this variant for a cleaner color inversion.
    size = int(rng.integers(3, 6))
    a, b = map(int, rng.choice(fg_colors, size=2, replace=False))
    tile = np.full((size, size), a, dtype=int)
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    # Paint several whole symmetry orbits with the second color so the pattern is structured but not
    # monochrome. The count range keeps both colors well represented.
    n_flip = int(rng.integers(1, max(2, len(orbits))))
    for orbit in orbits[:n_flip]:
        for r, c in orbit:
            tile[r, c] = b

    # Reject the rare case where orbit sizes made the tile monochrome or fully flipped.
    if np.all(tile == a) or np.all(tile == b):
        rr = int(rng.integers(0, size))
        cc = int(rng.integers(0, size))
        tile[rr, cc] = b if tile[rr, cc] == a else a
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
    while True:
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        area = h * w
        min_fill = max(2, int(np.ceil(0.40 * area)))
        max_fill = min(area - 2, int(np.floor(0.68 * area)))
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
            pick = frontier[int(rng.integers(len(frontier)))]
            mask[pick] = True

        fill = int(mask.sum())
        holes = area - fill
        if fill >= 2 and holes >= 2:
            ctrl = np.full((h, w), int(bg), dtype=int)
            ctrl[mask] = int(color)
            return ctrl


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    inv_tile = _swap_tile_colors(tile, bg)
    ctrl_color = int(colors_present(ctrl, exclude={bg})[0])
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = tile if ctrl[r, c] == ctrl_color else inv_tile
            paste(out, patch, r * th, c * tw)
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
    palette = list(rng.choice(np.arange(10, dtype=int), size=4, replace=False))
    bg = int(palette[0])
    tile_fg = [int(palette[1]), int(palette[2])]
    ctrl_color = int(palette[3])

    tile = _make_tile(bg, tile_fg, rng)
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

    if sorted(len(cols) for cols in colors_per_comp) != [1, 2]:
        return None

    # Anti-degeneration: the variant output must differ from the original blank-hole output.
    blank_hole_output = np.full_like(expected, bg)
    th, tw = tile.shape
    ctrl_fg = int(colors_present(ctrl, exclude={bg})[0])
    for r in range(ctrl.shape[0]):
        for c in range(ctrl.shape[1]):
            if ctrl[r, c] == ctrl_fg:
                paste(blank_hole_output, tile, r * th, c * tw)
    if np.array_equal(expected, blank_hole_output):
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
