# Combo variant of dbff022c (index 22)
# combo_spec: D1/0+D3/2+D5/0
# Name: 反向映射 + 反直觉方向 + 条纹填充

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - Baseline task: detect an edge palette, read it as border->fill, then flood-fill
#      each enclosed hole with one solid color derived from the surrounding border.
#    - Naive fusion risk:
#      If D1/0 only swaps palette semantics to fill->border and D3/2 only swaps
#      "outer row vs inner row", the two changes cancel visually in this task family:
#      the palette would again look like the original outer-border / inner-fill layout.
#      That would make the fusion degenerate, leaving only the striped fill visibly active.
#    - Resolution used here:
#      The palette still occupies a 2xN or Nx2 edge block, but the physically outer line
#      stores border colors while the physically inner line stores fill colors in reversed
#      legend order. The intended legend must therefore be read by pairing the outer line
#      with the reversed inner line. Logically the legend is fill->border; operationally
#      the solver inverts that logical map back to border->fill so it can color the holes.
#    - Small trace:
#      Suppose a top-edge palette shows outer row [2, 4, 6] and inner row [3, 1, 8].
#      Because the inner row is read from the opposite end, the logical pairs are
#      fill 8 -> border 2, fill 1 -> border 4, fill 3 -> border 6.
#      For a border-2 sprite, the correct fill is 8 while naive same-column reading
#      would claim 3. The middle slot of an odd-length palette would be a fixed point,
#      so the generator avoids using those center entries as active examples.
#      The hole is then filled as alternating stripes of 8 and background, not as a solid 8.
#    - Difference from original:
#      original = same-column direct mapping + solid flood fill
#      combo    = reversed pairing order + logically reversed mapping + striped writeback
#
# b) Complexity calibration
#    - Palette length stays small (2-4) so humans can infer pairings from 2-4 examples.
#    - Border colors and fill colors are chosen as disjoint sets to keep the reverse mapping
#      legible: fill colors appear only inside holes / on the inner legend line, while
#      border colors appear on sprite outlines / the outer legend line.
#    - Filled sprites are restricted to palette entries whose mirrored partner differs from
#      the same-column partner, so the reverse pairing rule is always behaviorally active.
#    - Stripe direction uses only two visible modes: wider-or-square holes -> vertical
#      stripes, taller holes -> horizontal stripes.
#
# c) Hidden assumption audit against the original main()
#    - Original _palette_mapping assumed outer line directly listed border keys and inner
#      line directly listed same-column fills. That assumption is false here: the inner line
#      must be read in reverse order, and the logical legend direction is fill->border.
#    - Original fill stage assumed a single solid writeback color. That breaks under D5/0,
#      because some cells of the hole must intentionally remain background.
#    - Original code safely collected all background components before mutating them. That
#      ordering is still required; otherwise newly written background stripe cells would
#      split holes mid-process.
#    - Original samples could tolerate palette entries whose same-column and mirrored
#      pairings accidentally agreed. This combo cannot, so the generator filters against
#      those fixed points.
#
# d) Placement legality
#    - Palette and sprites are separated by at least one background cell via the occupancy
#      mask. Sprites do not overlap each other or the palette padding.
#    - Holes are never allowed to touch the outer grid boundary.
#    - The palette block itself contains no background and is isolated from non-palette
#      cells with matching colors, so palette detection remains stable.
#
# e) Visual observability
#    - The reverse pairing is visible because used sprites never correspond to the inner-row
#      color in the same column; at least one active border maps to a mirrored-position fill.
#    - The reverse mapping is visible because the useful interior color set does not coincide
#      with the border-color set.
#    - The striped transform is visible because every filled sprite has at least one hole with
#      area >= 2, so the output differs from a solid fill.
#
# f) Visual cleanliness
#    - Sprites stay 1 cell apart and the palette stays compact on one edge.
#    - Interior stripes use only one fill color plus background, avoiding multi-layer clutter.
#    - The generator biases filled sprites to exhibit both stripe orientations across samples
#      when space permits, but never mixes multiple conflicting rules inside one hole.
#
# g) Change inferability
#    - The rule depends on only three repeated structural cues:
#      1. outlined hollow sprites,
#      2. an isolated edge legend with one line mirrored against the other,
#      3. a two-mode stripe pattern determined by hole aspect ratio.
#    - No hidden ordering uses numeric color ids or generation order. Only visible geometry
#      and visible legend arrangement drive the transformation.
#
# Post-change main() assumptions that the generator guarantees
#    - Every palette-hit hole is enclosed by a single dominant border color that exists in
#      the palette's border line.
#    - Every palette-hit hole has at least two cells, so stripe fill is observable.
#    - At least one used palette entry is non-fixed under the mirrored pairing rule, so the
#      reverse-direction cue changes the output compared with naive same-column reading.


# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 5, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7
_TINY_H_THRESHOLD = 5

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.15

_MAX_RETRIES = 500
# -- End configuration --------------------------------------------------------


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                    return False
        return True

    def consider_line_pair(side: str, outer: np.ndarray, inner: np.ndarray, limit: int) -> None:
        nonlocal best
        for s0 in range(limit):
            for s1 in range(s0 + 2, limit + 1):
                seg_outer = outer[s0:s1]
                seg_inner = inner[s0:s1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                if len(d_outer) != len([int(x) for x in seg_outer.tolist() if int(x) != bg]):
                    continue
                if len(d_inner) != len([int(x) for x in seg_inner.tolist() if int(x) != bg]):
                    continue
                if len(d_outer | d_inner) < 3:
                    continue
                if not isolated(side, s0, s1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (min(len(d_outer), len(d_inner)), len(d_outer | d_inner), -(s1 - s0), nonbg)
                cand = (score, side, s0, s1)
                if best is None or cand[0] > best[0]:
                    best = cand

    if h >= 2:
        consider_line_pair("top", g[0, :], g[1, :], w)
        consider_line_pair("bottom", g[h - 1, :], g[h - 2, :], w)
    if w >= 2:
        consider_line_pair("left", g[:, 0], g[:, 1], h)
        consider_line_pair("right", g[:, w - 1], g[:, w - 2], h)

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _extract_palette_lines(g: np.ndarray, side: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (outer_line, inner_line, palette_mask) for the chosen palette block."""
    h, w = g.shape
    pal_mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        pal_mask[0:2, start:end] = True
        outer_line = g[0, start:end]
        inner_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        outer_line = g[h - 1, start:end]
        inner_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        outer_line = g[start:end, 0]
        inner_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        outer_line = g[start:end, w - 1]
        inner_line = g[start:end, w - 2]
    return outer_line, inner_line, pal_mask


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Returns (mapping border->fill, palette_mask).

    The physical outer line lists border colors. The physical inner line lists fill colors,
    but that inner line must be read from the opposite end of the legend. The logical legend
    is therefore fill->border, and this function inverts it back to border->fill for use by
    the hole-filling stage.
    """
    side, start, end = _best_palette(g, bg)
    outer_line, inner_line, pal_mask = _extract_palette_lines(g, side, start, end)

    border_seq = [int(x) for x in outer_line.tolist() if int(x) != bg]
    fill_seq = [int(x) for x in inner_line[::-1].tolist() if int(x) != bg]
    assert len(border_seq) == len(fill_seq) and border_seq, "_palette_mapping expects a dense palette block"

    fill_to_border: dict[int, int] = {}
    for fill, border in zip(fill_seq, border_seq, strict=False):
        fill_to_border[int(fill)] = int(border)

    border_to_fill = {border: fill for fill, border in fill_to_border.items()}
    return border_to_fill, pal_mask


def _hole_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _stripe_axis_for_hole(pts: np.ndarray) -> str:
    r0, c0, r1, c1 = _hole_bbox(pts)
    hole_h = r1 - r0 + 1
    hole_w = c1 - c0 + 1
    return "vertical" if hole_w >= hole_h else "horizontal"


def _apply_striped_fill(out: np.ndarray, pts: np.ndarray, bg: int, fill: int) -> None:
    r0, c0, _, _ = _hole_bbox(pts)
    axis = _stripe_axis_for_hole(pts)
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        if axis == "horizontal":
            out[rr, cc] = int(fill) if (rr - r0) % 2 == 0 else int(bg)
        else:
            out[rr, cc] = int(fill) if (cc - c0) % 2 == 0 else int(bg)


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Find non-boundary background holes, infer the surrounding border color, and rewrite each
    hole as alternating fill/background stripes using the palette's mirrored reverse legend.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    comps = components_mask(bg_mask, connectivity=4)
    for pts in comps:
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
                    if v != bg:
                        adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col in mapping and cnt > best:
                best = cnt
                border = int(col)
        if border is None:
            continue

        _apply_striped_fill(out, np.asarray(pts), bg, int(mapping[border]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprite interiors using a reversed edge legend and alternating stripes.

    Input structure:
    - The dominant outer-border color is the background. Several hollow sprites sit away from the palette and from each other.
    - A compact 2xN or Nx2 palette block lies on one grid edge. Its edge-touching line shows border colors, while its inner line shows the fill colors, but that inner line must be read from the opposite end of the legend.

    Rule:
    - Detect the edge palette, pair the outer border-color line with the reversed inner fill-color line, and interpret those logical pairs as fill->border.
    - Invert that logical legend to recover which fill color belongs to each sprite border color.
    - For every enclosed background region that does not touch the grid boundary, identify the dominant adjacent border color and paint the hole with alternating stripes of the recovered fill color and background. Wider-or-square holes use vertical stripes; taller holes use horizontal stripes.

    Key insight:
    - The legend does not use the obvious same-column correspondence: one line is mirrored relative to the other, and the logical direction is reversed. After recovering the intended border/fill pairing, the interior transform is still not a solid fill but a geometry-driven stripe pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return local bg components not touching the sprite boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped boolean mask."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a hollow sprite with at least one enclosed hole and bounded area."""
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - n - 1)
        sprite[br, s : s + n] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - n - 1)
        sprite[s : s + n, bc] = True

    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(bcells)
        for cr, cc in bcells[: rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    holes = _get_enclosed_holes(sprite)
    if not holes or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _hole_visible_for_stripes(hpts: np.ndarray) -> bool:
    return int(len(hpts)) >= 2


def _supports_axis(mask: np.ndarray, axis: str) -> bool:
    for hpts in _get_enclosed_holes(mask):
        if _hole_visible_for_stripes(hpts) and _stripe_axis_for_hole(np.asarray(hpts)) == axis:
            return True
    return False


def _choose_colors(
    rng: random.Random, n_pairs: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """
    Choose disjoint border/fill palettes and force active sprites to use non-fixed mirrored entries.
    """
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n_pairs]
    fill_values = colors[n_pairs : 2 * n_pairs]
    distractor_pool = colors[2 * n_pairs :]
    if not distractor_pool:
        raise ValueError("_choose_colors requires at least one distractor color")

    active_idxs = [i for i in range(n_pairs) if i != n_pairs - 1 - i]
    if len(active_idxs) < num_filled:
        raise ValueError("not enough non-fixed palette entries for the requested filled sprites")
    chosen_idxs = rng.sample(active_idxs, num_filled)
    filled_keys = [border_keys[i] for i in chosen_idxs]

    mapping = {border: fill for border, fill in zip(border_keys, fill_values, strict=False)}
    distractor_color = distractor_pool[0]
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """
    Construct the physical legend.

    outer line: border colors in logical order
    inner line: fill colors in reversed logical order
    """
    n_pairs = len(border_keys)
    inner_physical = list(reversed(fill_values))
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_pairs), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = inner_physical
    else:
        pal = np.zeros((n_pairs, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = inner_physical
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> None:
    """Place palette on the chosen edge and mark its occupancy with 1-cell padding."""
    h, w = grid.shape
    ph, pw = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(0, w - pw)
    elif side == "bottom":
        pr0, pc0 = h - ph, rng.randint(0, w - pw)
    elif side == "left":
        pr0, pc0 = rng.randint(0, h - ph), 0
    else:
        pr0, pc0 = rng.randint(0, h - ph), w - pw
    grid[pr0 : pr0 + ph, pc0 : pc0 + pw] = pal
    occ[max(0, pr0 - 1) : min(h, pr0 + ph + 1), max(0, pc0 - 1) : min(w, pc0 + pw + 1)] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find all positions where the sprite fits and its holes stay away from the outer border."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + int(hr) == 0 or c0 + int(hc) == 0 or r0 + int(hr) == h - 1 or c0 + int(hc) == w - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_local_striped_hole(
    grid: np.ndarray, r0: int, c0: int, hpts: np.ndarray, bg: int, fill: int
) -> None:
    hr0, hc0, _, _ = _hole_bbox(np.asarray(hpts))
    axis = _stripe_axis_for_hole(np.asarray(hpts))
    for hr, hc in hpts:
        rr = r0 + int(hr)
        cc = c0 + int(hc)
        if axis == "horizontal":
            grid[rr, cc] = int(fill) if (int(hr) - hr0) % 2 == 0 else int(bg)
        else:
            grid[rr, cc] = int(fill) if (int(hc) - hc0) % 2 == 0 else int(bg)


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
    bg: int,
) -> None:
    """Render a sprite onto the grid; optionally paint striped holes."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
    if fill is not None:
        for hpts in holes:
            _paint_local_striped_hole(grid, r0, c0, np.asarray(hpts), bg, int(fill))


def _random_shape(
    rng: random.Random, max_sh: int, max_sw: int, allow_diamond: bool = True
) -> np.ndarray | None:
    sh = rng.randint(3, max_sh)
    sw = rng.randint(3, max_sw)
    if sh + sw > _SPRITE_DIM_SUM_CAP:
        sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
    if sw > max_sw:
        return None
    if sh == 3 and sw == 3 and allow_diamond and rng.random() < _P_DIAMOND:
        return _make_diamond()
    return _make_hollow_sprite(sh, sw, rng)


def _make_target_axes(rng: random.Random, num_filled: int) -> list[str]:
    if num_filled == 1:
        return [rng.choice(["horizontal", "vertical"])]
    axes = ["horizontal", "vertical"]
    while len(axes) < num_filled:
        axes.append(rng.choice(["horizontal", "vertical"]))
    rng.shuffle(axes)
    return axes


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    active_cap = len([i for i in range(n_pairs) if i != n_pairs - 1 - i])
    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, active_cap))
    total_sprites = num_filled + 1

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    target_axes = _make_target_axes(rng, num_filled)
    shapes: list[np.ndarray] = []
    for axis in target_axes:
        mask = None
        for _ in range(80):
            cand = _random_shape(rng, max_sh, max_sw, allow_diamond=False)
            if cand is not None and _supports_axis(cand, axis):
                mask = cand
                break
        if mask is None:
            return None
        shapes.append(mask)

    while len(shapes) < total_sprites:
        cand = _random_shape(rng, max_sh, max_sw, allow_diamond=True)
        if cand is None:
            return None
        if any(_hole_visible_for_stripes(hpts) for hpts in _get_enclosed_holes(cand)):
            shapes.append(cand)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n_pairs, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos: list[tuple[np.ndarray, int, int | None]] = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            color = filled_keys[i]
            fill = mapping[color]
        else:
            color = distractor_color
            fill = None
        infos.append((mask, color, fill))

    placed: list[tuple[int, int, np.ndarray, int | None]] = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        if fill is not None and not any(_hole_visible_for_stripes(hpts) for hpts in holes):
            return None
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes, bg)
        placed.append((r0, c0, mask, fill))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill in placed:
        if fill is None:
            continue
        for hpts in _get_enclosed_holes(mask):
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if np.array_equal(inp, output):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
