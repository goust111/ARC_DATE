# Variant of bf45cf4b
# Dimension: D11 — 源-目标基数关系 (meta)
# Change: 2个输入对象（1个控制精灵 + 1个瓦片）→ 1个输出图案（1:1映射中的2:1聚合）... → 1个输入对象自我放大（1:1）

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


# Design analysis for the D11 variant
# ==================================
# Source-task note:
# - The referenced task b99e7126 is present in this workspace only as evaluation JSON.
#   Its solver source is not available locally, so the "fractal self-similarity" idea is
#   used only as a behavioral baseline, per the instruction to skip missing source solvers.
#
# a) Concrete behavior and flow change
# - Original bf45cf4b:
#   input = 2 disconnected objects.
#   One object is a monochrome control mask with holes; the other is a multicolor tile.
#   Output stamps the tile once for every filled control pixel.
# - D11 variant:
#   input = exactly 1 connected object S.
#   Crop S to its tight bbox. Inside that bbox:
#   1. The non-background occupancy pattern of S is the control mask.
#   2. The full colored patch S itself is the tile.
#   Therefore the object is self-referential: each filled pixel of S expands into a copy of S.
# - Small example:
#     S =
#       a b .
#       a c b
#       . c .
#   Original logic would need a separate control and a separate block.
#   Variant logic uses mask(S != bg) and block=S, so the output is a 3x3 tiling lattice
#   where five macro-cells contain the exact 3x3 pattern S and four macro-cells stay bg.
#
# b) Complexity calibration
# - A fully filled rectangle would make the rule visually weaker, so the generator forces
#   at least one background hole inside the bbox.
# - A monochrome sprite would collapse the "tile" aspect into a plain mask, so the sprite
#   always uses 2-3 foreground colors.
# - The bbox is kept at 3-5 by 3-5 so the output size is 9-25 in each dimension:
#   large enough for visible self-similarity, but still compact and easy to parse.
#
# c) Implicit assumption review against the original main()
# - `len(comps) == 2` depended on the old 2-object layout. It must become `len(comps) == 1`.
# - Sorting components by color count / squareness assumed a separate block existed.
#   That distinction disappears entirely.
# - `ctrl_cols[0]` assumed the control sprite was monochrome. This is invalid now because
#   the single object is multicolor. The correct mask is simply `sprite != bg`.
# - `block` and `ctrl` had separate shapes before. Now both roles point to the same bbox patch,
#   so the output size becomes `(sprite_h * sprite_h) x (sprite_w * sprite_w)`.
#
# d) Placement legality
# - The generator places one sprite only, so overlap/separation issues from the old generator
#   disappear.
# - The sprite is pasted wholly inside a 12-18 by 12-18 grid with random slack; bg remains
#   the most common color because the sprite bbox is at most 5x5.
# - The sprite mask is grown by 8-neighbor frontier expansion from a cross seed, guaranteeing
#   one connected foreground component and at least one filled cell in every row and column.
#
# e) Visual observability
# - Because the input already shows the full multicolor motif, the output visibly reuses that
#   same motif as a repeated miniature. This makes the self-reference inferable from 2-3 pairs.
# - Multiple colors are preserved exactly; there is no hidden recoloring map to guess.
#
# f) Visual cleanliness
# - There is only one object in the input, with no stacked annotations or overlapping markers.
# - Inside the output, copies occupy non-overlapping macro-cells on a clean background lattice.
#
# g) Color inference
# - Colors are not transformed. Every output tile is an exact copy of the input sprite patch.
# - The generator limits the sprite to 2-3 foreground colors so repetition is visible without
#   becoming noisy.
#
# h) Variant audit metric
# - Core metric: self-scale factor = sprite_area = sprite_h * sprite_w.
#   In this variant it ranges from 9 to 25 and directly controls the output expansion.
# - Supporting metrics: number of foreground colors, hole count, and the invariant that the
#   input always has exactly one 8-connected foreground component.
#
# i) New solver assumptions and generator guarantees
# - main() assumes a unique foreground component exists. generate_input() verifies this.
# - main() assumes the component bbox is the intended sprite bbox. The mask generator forces
#   at least one filled cell in every row and every column, so cropping does not shrink it.
# - main() treats every non-bg cell as a stamp location. The generator ensures the object is
#   connected and holey, so the task does not degenerate into a plain full-rectangle scaling.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Self-scale a single multicolor sprite by using the sprite itself as both mask and tile.

    Input structure:
    - bg = background color.
    - Exactly one 8-connected foreground component.
    - Its bbox patch `sprite` may contain multiple colors and background holes.

    Rule:
    - Let `mask = (sprite != bg)`.
    - For every filled cell in `mask`, stamp a full copy of `sprite`.
    - Leave background-sized gaps where `mask` is false.

    Output size is `(sprite_h * sprite_h) x (sprite_w * sprite_w)`.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 1

    sprite = crop_bbox(g, comps[0]["bbox"])
    sh, sw = map(int, sprite.shape)
    mask = (sprite != bg)
    if not mask.any():
        return np.full((0, 0), int(bg), dtype=int)

    out = np.full((sh * sh, sw * sw), int(bg), dtype=int)
    for r in range(sh):
        for c in range(sw):
            if bool(mask[r, c]):
                paste(out, sprite, r * sh, c * sw)
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


def _make_mask(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    area = h * w

    seed_r = int(rng.integers(0, h))
    seed_c = int(rng.integers(0, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, seed_c] = True
    mask[seed_r, :] = True

    min_fill = max(int(mask.sum()), int(np.ceil(0.45 * area)))
    max_fill = max(min_fill, int(np.floor(0.76 * area)))
    if max_fill >= area:
        max_fill = area - 1
    target = int(rng.integers(min_fill, max_fill + 1))

    while int(mask.sum()) < target:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                    frontier.append((r, c))
        assert frontier
        pick_r, pick_c = frontier[int(rng.integers(len(frontier)))]
        mask[pick_r, pick_c] = True

    return mask


def _coord_key(mode: str, anchor: tuple[int, int], coord: tuple[int, int]) -> tuple[int, int, int]:
    r, c = coord
    ar, ac = anchor
    if mode == "col":
        return c, r, 0
    if mode == "diag":
        return r + c, r, c
    if mode == "anti":
        return r - c, r, c
    if mode == "center":
        return abs(r - ar) + abs(c - ac), r, c
    return r, c, 0


def _paint_sprite(mask: np.ndarray, bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    h, w = mask.shape
    sprite = np.full((h, w), int(bg), dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]

    n_colors = int(rng.integers(2, min(3, len(coords)) + 1))
    palette = [int(x) for x in rng.choice(fg_colors, size=n_colors, replace=False)]

    anchor = coords[int(rng.integers(len(coords)))]
    mode = str(rng.choice(["row", "col", "diag", "anti", "center"]))
    ordered = sorted(coords, key=lambda rc: _coord_key(mode, anchor, rc))
    for idx, (r, c) in enumerate(ordered):
        sprite[r, c] = int(palette[idx % n_colors])

    # Ensure all selected colors are visibly used, even on tiny masks.
    for idx, color in enumerate(palette):
        r, c = ordered[idx]
        sprite[r, c] = int(color)

    used = colors_present(sprite, exclude={bg})
    assert len(used) >= 2
    return sprite


def _compose_output(sprite: np.ndarray, bg: int) -> np.ndarray:
    sh, sw = sprite.shape
    out = np.full((sh * sh, sw * sw), int(bg), dtype=int)
    for r in range(sh):
        for c in range(sw):
            if sprite[r, c] != bg:
                paste(out, sprite, r * sh, c * sw)
    return out


def _make_input(sprite: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    sh, sw = sprite.shape
    H = int(rng.integers(max(12, sh + 2), 19))
    W = int(rng.integers(max(12, sw + 2), 19))
    r0 = int(rng.integers(0, H - sh + 1))
    c0 = int(rng.integers(0, W - sw + 1))

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, sprite, r0, c0, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=4, replace=False)]
    bg = int(palette[0])
    fg = palette[1:]

    mask = _make_mask(rng)
    sprite = _paint_sprite(mask, bg, fg, rng)
    inp = _make_input(sprite, bg, rng)
    expected = _compose_output(sprite, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 1:
        return None

    patch = crop_bbox(inp, comps[0]["bbox"])
    if patch.shape != sprite.shape:
        return None
    if not np.array_equal(patch, sprite):
        return None
    if len(colors_present(patch, exclude={bg})) < 2:
        return None
    if int(np.sum(patch == bg)) == 0:
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
