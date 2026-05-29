# Combo variant of dbff022c (index 30)
# combo_spec: D1/2+D11/2
# Name: 多对一映射 + 精灵网格排列

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original samples scatter a few hollow sprites and rely on a one-to-one palette.
#    Here the palette still sits on an edge, but several distinct border colors can point
#    to the same fill color. The sprites are no longer scattered: they occupy a regular
#    3x3 lattice of equal slots. Example: border keys [1, 2, 3, 4] may map to fills
#    [7, 7, 8, 8]. In the input, nine hollow boxes sit in a checkerboard arrangement.
#    In the output, every box with border 1 or 2 gets filled with 7, and every box with
#    border 3 or 4 gets filled with 8. Relative to the original, the changed steps are:
#    palette reading now permits repeated fill values, and sprite placement is fixed to a
#    row/column grid so the repeated fill grouping becomes visually obvious.
# b) Complexity calibration:
#    The layout is regular but not trivial: 9 sprites force repeated evidence for the same
#    mapping rule, while only 4-5 border keys and exactly 2 fill groups keep the color
#    vocabulary small enough for a human to infer. Using one repeated base shape prevents
#    visual clutter and keeps the main challenge on palette decoding plus many-to-one reuse.
# c) Broken implicit assumptions from the original code:
#    - _best_palette previously preferred at least two distinct colors on both palette lines.
#      That can reject valid many-to-one legends where the fill line repeats heavily.
#      The solver below only requires the outer border-key line to be distinct; the inner
#      fill line may repeat.
#    - mapping[b] = f remains valid because border keys are still unique, even though fill
#      values are not.
#    - Hole filling still assumes each enclosed background component is dominated by a
#      single adjacent border color. The generator preserves that by using one-color hollow
#      sprites with 1-cell gutters between all lattice slots.
# d) Placement legality:
#    The generator reserves a palette band on one edge, inserts a 1-cell gap, and then
#    paints the 3x3 sprite lattice inside the remaining area with 1-cell gutters. This
#    guarantees no overlap, no out-of-bounds placement, and no hole touching the grid edge.
# e) Visual observability:
#    A solver can see repeated same-shape sprites on a predictable grid, then compare
#    different border colors that produce the same interior fill after transformation.
#    The repeated checkerboard grouping gives multiple visible witnesses of the many-to-one
#    rule in a single example.
# f) Visual cleanliness:
#    Only three element types share the board: palette, hollow sprites, and background.
#    They are separated spatially, so no region is overloaded with multiple annotations.
# g) Change inferability:
#    The palette provides the only color rule, and only two output fill colors are used.
#    That keeps the transformation legible across 2-4 ARC examples instead of turning it
#    into an arbitrary recoloring puzzle.
# h) New solver assumptions after modification:
#    - The palette is a contiguous 2xN or Nx2 edge block with a fully non-background outer
#      key line and a fully non-background inner fill line.
#    - Sprites and palette are separated by at least one background cell so palette
#      isolation is detectable.
#    - Every enclosed hole belongs to a single-color border whose color is either present
#      in the palette or left unchanged.
#    The generator constructs inputs to satisfy all three conditions.


def _bg_boundary(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0:2, start:end] = True
    elif side == "bottom":
        mask[h - 2 : h, start:end] = True
    elif side == "left":
        mask[start:end, 0:2] = True
    else:
        mask[start:end, w - 2 : w] = True
    return mask


def _edge_lines(g: np.ndarray, side: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = g.shape
    if side == "top":
        return g[0, start:end], g[1, start:end]
    if side == "bottom":
        return g[h - 1, start:end], g[h - 2, start:end]
    if side == "left":
        return g[start:end, 0], g[start:end, 1]
    return g[start:end, w - 1], g[start:end, w - 2]


def _is_isolated_palette(g: np.ndarray, bg: int, pal_mask: np.ndarray) -> bool:
    h, w = g.shape
    for r, c in zip(*np.where(pal_mask), strict=False):
        v = int(g[int(r), int(c)])
        if v == bg:
            continue
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                if int(g[rr, cc]) == v:
                    return False
    return True


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on an edge.

    The key change from the original task is that the inner fill line may repeat colors.
    Only the outer border-key line must stay distinct.
    """
    h, w = g.shape
    best = None

    def consider(side: str, limit: int) -> None:
        nonlocal best
        for start in range(limit):
            for end in range(start + 2, limit + 1):
                outer, inner = _edge_lines(g, side, start, end)
                outer_vals = [int(x) for x in outer.tolist()]
                inner_vals = [int(x) for x in inner.tolist()]
                if any(v == bg for v in outer_vals + inner_vals):
                    continue
                if len(set(outer_vals)) != len(outer_vals):
                    continue
                if len(outer_vals) < 2 or len(set(inner_vals)) < 1:
                    continue
                pal_mask = _palette_mask(g.shape, side, start, end)
                if not _is_isolated_palette(g, bg, pal_mask):
                    continue
                score = (len(outer_vals), len(set(inner_vals)))
                cand = (score, side, start, end)
                if best is None or cand[0] > best[0]:
                    best = cand

    if h >= 2:
        consider("top", w)
        consider("bottom", w)
    if w >= 2:
        consider("left", h)
        consider("right", h)

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    side, start, end = _best_palette(g, bg)
    pal_mask = _palette_mask(g.shape, side, start, end)
    border_line, fill_line = _edge_lines(g, side, start, end)

    mapping: dict[int, int] = {}
    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b != bg:
            mapping[b] = f
    return mapping, pal_mask


def _fill_enclosed_bg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        adj_counts: Counter[int] = Counter()
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
                    if v != bg:
                        adj_counts[v] += 1

        border = None
        best = -1
        for color, cnt in adj_counts.items():
            if color in mapping and cnt > best:
                border = color
                best = cnt
        if border is None:
            continue

        fill = int(mapping[border])
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the interiors of regularly arranged hollow sprites using an edge palette legend,
    even when several border colors share the same target fill color.

    Input structure:
    - The background is the dominant border color.
    - One edge contains a compact 2xN or Nx2 palette whose outer line lists distinct
      sprite-border colors and whose inner line lists the fill colors paired to them.
      Different border colors may point to the same fill color.
    - Away from the palette, many hollow sprites are placed on a regular row/column grid.

    Rule:
    - Detect the palette on the grid edge and read each aligned pair as border color ->
      fill color.
    - Find every enclosed background region that is not part of the palette and does not
      touch the board boundary.
    - For each such region, inspect the surrounding border color, look it up in the
      palette, and replace the region with the mapped fill value.

    Key insight:
    - The edge legend is still the only source of recoloring information, but the fill line
      is no longer injective: several different border colors can collapse to the same fill.
      The regular sprite lattice provides repeated visual evidence of those shared outcomes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator ---------------

_MAX_RETRIES = 200
_PALETTE_LEN_CHOICES = (4, 5)
_LAYOUT_ROWS = 3
_LAYOUT_COLS = 3
_SPRITE_SHAPES = ((4, 4), (4, 5), (5, 4))
_GAP = 1
_MARGIN = 1
_PALETTE_THICKNESS = 2


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_base_sprite(sh: int, sw: int, rng: random.Random) -> np.ndarray:
    sprite = np.ones((sh, sw), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if sh >= 5 and rng.random() < 0.5:
        sprite[sh // 2, 1 : sw - 1] = True
    elif sw >= 5 and rng.random() < 0.5:
        sprite[1 : sh - 1, sw // 2] = True

    holes = _get_enclosed_holes(sprite)
    assert holes, "_make_base_sprite must produce at least one enclosed hole"
    return sprite


def _choose_colors(rng: random.Random, n_keys: int) -> tuple[list[int], dict[int, int], list[list[int]]]:
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n_keys]
    fill_a, fill_b = colors[n_keys : n_keys + 2]

    first_group_size = 2 if n_keys == 4 else 3
    key_groups = [border_keys[:first_group_size], border_keys[first_group_size:]]

    mapping: dict[int, int] = {}
    for key in key_groups[0]:
        mapping[key] = fill_a
    for key in key_groups[1]:
        mapping[key] = fill_b
    return border_keys, mapping, key_groups


def _build_palette(side: str, border_keys: list[int], mapping: dict[int, int]) -> np.ndarray:
    fill_values = [mapping[k] for k in border_keys]
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _grid_size_and_origins(
    side: str, sh: int, sw: int, pal_len: int
) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    lattice_h = _LAYOUT_ROWS * sh + (_LAYOUT_ROWS - 1) * _GAP
    lattice_w = _LAYOUT_COLS * sw + (_LAYOUT_COLS - 1) * _GAP

    if side in ("top", "bottom"):
        h = _PALETTE_THICKNESS + _GAP + lattice_h + _MARGIN
        w = lattice_w + 2 * _MARGIN + 1
        sprite_origin = (
            (_PALETTE_THICKNESS + _GAP if side == "top" else _MARGIN),
            _MARGIN,
        )
        palette_origin = (0, 0) if side == "top" else (h - _PALETTE_THICKNESS, 0)
    else:
        h = lattice_h + 2 * _MARGIN + 1
        w = _PALETTE_THICKNESS + _GAP + lattice_w + _MARGIN
        sprite_origin = (
            _MARGIN,
            (_PALETTE_THICKNESS + _GAP if side == "left" else _MARGIN),
        )
        palette_origin = (0, 0) if side == "left" else (0, w - _PALETTE_THICKNESS)

    if side in ("top", "bottom"):
        assert w >= pal_len
    else:
        assert h >= pal_len
    return h, w, sprite_origin, palette_origin


def _place_palette(grid: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> np.ndarray:
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    pal_mask = np.zeros((h, w), dtype=bool)

    if side == "top":
        pr0, pc0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        pr0, pc0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        pr0, pc0 = rng.randint(0, h - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, h - pal_h), w - pal_w

    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    pal_mask[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = True
    return pal_mask


def _slot_positions(sprite_origin: tuple[int, int], sh: int, sw: int) -> list[tuple[int, int]]:
    sr0, sc0 = sprite_origin
    return [
        (sr0 + r * (sh + _GAP), sc0 + c * (sw + _GAP))
        for r in range(_LAYOUT_ROWS)
        for c in range(_LAYOUT_COLS)
    ]


def _paint_sprite(
    grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int, fill: int | None
) -> None:
    sh, sw = mask.shape
    grid[r0 : r0 + sh, c0 : c0 + sw][mask] = color
    if fill is not None:
        for hole in _get_enclosed_holes(mask):
            for hr, hc in hole:
                grid[r0 + int(hr), c0 + int(hc)] = int(fill)


def _assignment_pattern(group_keys: list[list[int]]) -> list[int]:
    groups = []
    for r in range(_LAYOUT_ROWS):
        for c in range(_LAYOUT_COLS):
            groups.append((r + c) % 2)
    return groups


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    side = rng.choice(["top", "bottom", "left", "right"])
    sh, sw = rng.choice(_SPRITE_SHAPES)
    sprite = _make_base_sprite(sh, sw, rng)

    pal_len = rng.choice(_PALETTE_LEN_CHOICES)
    border_keys, mapping, key_groups = _choose_colors(rng, pal_len)
    pal = _build_palette(side, border_keys, mapping)

    h, w, sprite_origin, _ = _grid_size_and_origins(side, sh, sw, pal_len)
    output = np.full((h, w), bg, dtype=int)
    pal_mask = _place_palette(output, pal, side, rng)

    positions = _slot_positions(sprite_origin, sh, sw)
    group_pattern = _assignment_pattern(key_groups)
    group_iters = [0, 0]

    placed = []
    for idx, (r0, c0) in enumerate(positions):
        group_id = group_pattern[idx]
        keys = key_groups[group_id]
        color = keys[group_iters[group_id] % len(keys)]
        group_iters[group_id] += 1
        fill = mapping[color]
        _paint_sprite(output, r0, c0, sprite, color, fill)
        placed.append((r0, c0, fill))

    inp = output.copy()
    for r0, c0, _fill in placed:
        for hole in _get_enclosed_holes(sprite):
            for hr, hc in hole:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if not np.array_equal(main(inp), output):
        return None

    # Generator-side fusion guards: repeated fills and regular 3x3 lattice must both hold.
    if len(set(mapping.values())) >= len(mapping):
        return None
    if pal_mask.sum() == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
