# Combo variant of bf45cf4b (index 15)
# combo_spec: D11/0+D9/1
# Name: 分形递归自放大

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
# - Original bf45cf4b:
#   input has 2 disconnected objects: monochrome control C and colored tile T.
#   Output = expand(mask(C), tile=T).
# - D11/0 removes the 2:1 aggregation:
#   there is only one object S, and both roles become self-referential.
#   stage1 = expand(mask(S), tile=S).
# - D9/1 adds one chained step. In the fused variant there is still only one input object,
#   so the natural readable recursion is:
#       mask0  = (S != bg)
#       stage1 = expand(mask0, tile=S)
#       final  = expand(mask(stage1), tile=S)
#   This means every filled pixel of the first enlarged result is replaced once more by S.
# - Small example:
#     S =
#       a b .
#       a c b
#       . c .
#   Original task: impossible without a separate control and tile.
#   D11 only: each of the 5 filled cells becomes one 3x3 copy of S, giving a 9x9 result.
#   Combo: after that, every filled cell inside the 9x9 stage1 becomes S again, producing
#   a 27x27 self-similar pattern with two visible nesting levels.
# - Using `expand(mask(S), tile=stage1)` would produce the same array because macro-tiling by
#   mask(S) and per-pixel substitution by S are associative here; the second phrasing is closer
#   to the requested "对结果中每个原始像素重复此操作".
#
# b) Complexity calibration
# - A fully filled rectangle would make the second stage look like a plain block scaling, so the
#   generator forces at least one background hole inside the bbox.
# - A monochrome sprite would hide the "tile equals self" idea, so the sprite always uses 2-3
#   foreground colors.
# - The sprite bbox is restricted to 3-4 by 3-4. Final outputs are therefore 27-64 per axis:
#   large enough to show recursion clearly, but still cheap to generate and visually parse.
# - The rule space is intentionally narrow: exactly one component, one mask, and exactly 2 rounds.
#
# c) Implicit assumption review against the original main()
# - `len(comps) == 2` is no longer valid; the fused task must require exactly 1 component.
# - Sorting components into "block" vs "control" disappears entirely.
# - `ctrl_cols[0]` relied on a monochrome control object; this is invalid because the lone sprite
#   is multicolor. The only correct mask is `sprite != bg`.
# - Original output size depended on independent control/tile shapes. Here all scales derive from
#   the same sprite: stage1 has shape `(h*h, w*w)` and final has shape `(h*h*h, w*w*w)`.
# - The solver assumes the component bbox is the intended sprite bbox. The generator guarantees
#   that every row and every column contains at least one filled cell, so cropping is stable.
#
# d) Placement legality
# - There is only one foreground object, so overlap/separation problems from the source task vanish.
# - The generator pastes the sprite fully inside a 12-18 by 12-18 canvas with random slack.
# - Because the sprite is at most 4x4, the background remains the most common color.
#
# e) Visual observability
# - The exact input motif reappears at two nested scales in the output, which is directly visible
#   from a few ARC examples.
# - No hidden dependence on numeric color ids or arbitrary ordering is used. Colors are preserved.
#
# f) Visual cleanliness
# - The input contains one isolated sprite on a blank background.
# - The output is a regular lattice of non-overlapping copies; there are no extra marks or overlays.
#
# g) Change inferability
# - There is no recoloring map to infer. The only change is recursive spatial substitution.
# - Restricting the sprite to 2-3 colors and requiring holes keeps the motif legible at both levels.
#
# h) Fusion audit metric
# - Let n = number of filled pixels in S. Then:
#       filled(stage1) = n^2
#       filled(final)  = n^3
#   The combo-specific signature is therefore:
#       recursion_gain = filled(final) / filled(stage1) = n
#   The generator enforces n >= 5, so the second stage always adds real structure and never
#   collapses to the single-stage D11 output.
#
# i) New solver assumptions and generator guarantees
# - main() assumes exactly one 8-connected foreground component; generate_input() verifies this.
# - main() assumes the sprite has at least one hole and at least 2 foreground colors; generation
#   constructs and validates both properties.
# - main() assumes the recursive expansion is computationally bounded; the 4x4 sprite cap ensures
#   that the largest output is 64x64.


def _expand_with_mask(tile: np.ndarray, mask: np.ndarray, bg: int) -> np.ndarray:
    tile = np.array(tile, dtype=int)
    mask = np.array(mask, dtype=bool)
    th, tw = map(int, tile.shape)
    mh, mw = map(int, mask.shape)
    out = np.full((mh * th, mw * tw), int(bg), dtype=int)
    for r in range(mh):
        for c in range(mw):
            if bool(mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply two rounds of self-referential scaling to one multicolor sprite.

    Input structure:
    - bg = most common color.
    - Exactly one 8-connected foreground component.
    - Its cropped patch `sprite` may contain multiple colors and background holes.

    Rule:
    1. Use occupancy of `sprite` as a binary control mask.
    2. Stamp the full colored `sprite` at every filled location once.
    3. Treat every filled pixel of that first enlarged result as another stamp location.
    4. Stamp `sprite` again there and return the tight result.

    Output size is `(sprite_h**3, sprite_w**3)`.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 1

    sprite = crop_bbox(g, comps[0]["bbox"])
    mask0 = (sprite != bg)
    if not bool(mask0.any()):
        return np.full((0, 0), int(bg), dtype=int)

    stage1 = _expand_with_mask(sprite, mask0, bg)
    final = _expand_with_mask(sprite, stage1 != bg, bg)
    return final


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
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
    h = int(rng.integers(3, 5))
    w = int(rng.integers(3, 5))
    area = h * w

    seed_r = int(rng.integers(0, h))
    seed_c = int(rng.integers(0, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, seed_c] = True
    mask[seed_r, :] = True

    min_fill = max(int(mask.sum()), int(np.ceil(0.45 * area)))
    max_fill = max(min_fill, int(np.floor(0.72 * area)))
    if max_fill >= area:
        max_fill = area - 1
    target = int(rng.integers(min_fill, max_fill + 1))

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                    frontier.append((r, c))
        assert frontier
        pick_r, pick_c = frontier[int(rng.integers(len(frontier)))]
        mask[pick_r, pick_c] = True

    assert 5 <= int(mask.sum()) < area
    assert np.all(mask.any(axis=1))
    assert np.all(mask.any(axis=0))
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

    for idx, color in enumerate(palette):
        r, c = ordered[idx]
        sprite[r, c] = int(color)

    assert len(colors_present(sprite, exclude={bg})) >= 2
    return sprite


def _compose_output(sprite: np.ndarray, bg: int) -> np.ndarray:
    stage1 = _expand_with_mask(sprite, sprite != bg, bg)
    return _expand_with_mask(sprite, stage1 != bg, bg)


def _make_input(sprite: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    sh, sw = sprite.shape
    H = int(rng.integers(max(12, sh + 2), 19))
    W = int(rng.integers(max(12, sw + 2), 19))
    r0 = int(rng.integers(0, H - sh + 1))
    c0 = int(rng.integers(0, W - sw + 1))

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, sprite, r0, c0, transparent=bg)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=4, replace=False)]
    bg = int(palette[0])
    fg = palette[1:]

    mask = _make_mask(rng)
    sprite = _paint_sprite(mask, bg, fg, rng)
    inp = _make_input(sprite, bg, rng)
    stage1 = _expand_with_mask(sprite, sprite != bg, bg)
    expected = _expand_with_mask(sprite, stage1 != bg, bg)
    single_step = stage1

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

    fill = int(np.sum(sprite != bg))
    if expected.shape == single_step.shape:
        return None
    if np.array_equal(expected, single_step):
        return None
    if expected.shape != (sprite.shape[0] ** 3, sprite.shape[1] ** 3):
        return None
    if int(np.sum(single_step != bg)) != fill * fill:
        return None
    if int(np.sum(expected != bg)) != fill * fill * fill:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
