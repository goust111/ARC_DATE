# Combo variant of dbff022c (index 17)
# combo_spec: D5/0+D6/2
# Name: 条纹填充 + 精灵单独裁剪排列输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis
# ---
# a) Concretization and behavior trace
#    - Original task:
#      detect the edge palette, use it to map border color -> interior fill color,
#      flood-fill every matched enclosed hole in place, and keep the full input-sized
#      canvas as output.
#    - Fused task here:
#      the palette logic is unchanged, but every matched hole becomes alternating
#      stripes instead of a solid fill, and then the whole scene is discarded except
#      for the matched sprites themselves. Each matched sprite is cropped to its own
#      tight bbox and the cropped results are arranged left-to-right in reading order.
#    - Concrete trace:
#      suppose a 5x6 sprite with border color 4 has a 3x2 hole and palette entry
#      4 -> 7. The original output would keep the sprite at its source position and
#      turn the full 3x2 hole into 7. The fused output first inspects the hole bbox:
#      height 3 > width 2, so it uses horizontal stripes, meaning local rows 0 and 2
#      become 7 while local row 1 stays background. Then this filled sprite is cropped
#      to its 5x6 bbox and placed as one item in a packed strip, with no palette and
#      no distractor sprites in the final output.
#    - Net difference:
#      D5 changes the local writeback pattern inside each hole; D6 changes the global
#      coordinate system of the answer. The solver must therefore finish the semantic
#      fill first and only then convert matched sprites into packed object outputs.
# b) Complexity calibration
#    - Stripe direction is intentionally limited to two visible cases:
#      wider-or-square holes -> vertical stripes, taller holes -> horizontal stripes.
#      This keeps the rule learnable from a few examples.
#    - Packed layout is intentionally limited to one horizontal strip with 1-cell
#      background gaps. Because there are only 1-3 matched sprites and each bbox is
#      at most 7x6, the strip stays compact while still being visibly different from
#      the original full-scene canvas.
#    - The generator forces every matched sprite to contain at least one enclosed hole
#      of area >= 2, so the striped rule cannot silently collapse back to solid fill.
# c) Hidden-assumption audit of the original solver
#    - `_best_palette()` and `_palette_mapping()` assume the palette remains an isolated
#      2xN or Nx2 legend on the outer border. This still holds.
#    - The original fill routine assumed each hole could be overwritten uniformly.
#      That assumption is broken; we now need hole-local bbox parity to decide which
#      cells become fill and which stay background.
#    - The original output logic implicitly assumed the solved scene itself was the
#      answer. D6 breaks that assumption; after solving the scene, we must identify
#      only the palette-hit sprites and repack those cropped pieces into a new canvas.
#    - Extracting packed sprites introduces a new assumption: every matched sprite is
#      one 4-connected non-bg component once stripes are written. The generator now
#      guarantees connected sprite borders so this is safe.
# d) Placement legality
#    - Input generation keeps the palette and all sprites separated by at least one
#      background cell, exactly as in the source task, so palette detection and hole
#      detection remain unambiguous.
#    - Output packing never overlaps items: each cropped sprite is placed into a fresh
#      horizontal strip with a single background column between consecutive pieces.
# e) Visual observability
#    - The palette still explains color choice. The new fill rule is visible because
#      matched interiors retain alternating background bands instead of becoming solid.
#    - The packed output is also immediately visible: the palette disappears, distractor
#      sprites disappear, and only solved matched sprites survive in reading order.
#    - No branch depends on hidden properties such as numeric color rank; stripe axis
#      comes from visible hole geometry and piece order comes from visible top-left
#      reading order in the input scene.
# f) Visual cleanliness
#    - Input semantics remain cleanly separated: one edge legend plus isolated hollow
#      sprites on background.
#    - Output semantics are even cleaner: a single strip of cropped solved sprites with
#      uniform spacing and no unrelated scene clutter.
# g) Change inferability
#    - The only color transform is still palette-mediated and therefore externally
#      signaled. The extra variation is structural: stripe orientation follows hole
#      aspect ratio, and packing follows source reading order. Both are repeatable cues
#      that can be inferred from 2-4 examples.
# h) New solver assumptions introduced here
#    - Every matched sprite must have at least one visible stripe-producing hole.
#      Generator guarantee: all palette-hit sprites have an enclosed hole with area >= 2.
#    - Multiple matched holes may belong to one sprite, so the solver deduplicates by
#      post-fill component id before packing.


# ----------------------------- solver helpers ------------------------------


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top", "bottom", "left", "right"}
      - for top/bottom: start/end are column indices [start, end)
      - for left/right: start/end are row indices [start, end)
    """
    h, w = g.shape
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        """Palette colors should not 4-touch same-colored cells outside the legend."""
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
            rr = int(r)
            cc = int(c)
            val = int(g[rr, cc])
            if val == bg:
                continue
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and not pal[nr, nc]:
                    if int(g[nr, nc]) == val:
                        return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[0, :]
        inner = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                merged = d_outer | d_inner
                if len(merged) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(merged), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[h - 1, :]
        inner = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                merged = d_outer | d_inner
                if len(merged) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(merged), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, 0]
        inner = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                merged = d_outer | d_inner
                if len(merged) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(merged), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, w - 1]
        inner = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                merged = d_outer | d_inner
                if len(merged) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(merged), -(r1 - r0), nonbg), "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return (mapping border->fill, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        border = int(b)
        fill = int(f)
        if border == bg:
            continue
        mapping[border] = fill

    return mapping, pal_mask


def _bbox_from_pts(pts: np.ndarray) -> tuple[int, int, int, int]:
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _stripe_axis_for_hole(pts: np.ndarray) -> str:
    r0, c0, r1, c1 = _bbox_from_pts(pts)
    hole_h = int(r1 - r0 + 1)
    hole_w = int(c1 - c0 + 1)
    return "vertical" if hole_w >= hole_h else "horizontal"


def _paint_striped_hole(out: np.ndarray, pts: np.ndarray, bg: int, fill: int) -> None:
    r0, c0, _, _ = _bbox_from_pts(pts)
    axis = _stripe_axis_for_hole(pts)
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        if axis == "horizontal":
            out[rr, cc] = int(fill) if (rr - r0) % 2 == 0 else int(bg)
        else:
            out[rr, cc] = int(fill) if (cc - c0) % 2 == 0 else int(bg)


def _matched_hole_infos(
    g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> list[dict[str, object]]:
    """
    Find enclosed bg holes that map to palette keys.

    Each record stores:
      - pts: coords of the hole
      - fill: mapped fill color
      - seed: one adjacent border cell of the chosen sprite
    """
    h, w = g.shape
    infos: list[dict[str, object]] = []
    bg_mask = (g == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        adj_counts: dict[int, int] = {}
        seeds: dict[int, tuple[int, int]] = {}
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            for nr, nc in neighbors4(rr, cc):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if pal_mask[nr, nc]:
                    continue
                val = int(g[nr, nc])
                if val == bg:
                    continue
                adj_counts[val] = adj_counts.get(val, 0) + 1
                seeds.setdefault(val, (int(nr), int(nc)))

        best_border = None
        best_count = -1
        for color, cnt in adj_counts.items():
            if color not in mapping:
                continue
            if cnt > best_count:
                best_count = int(cnt)
                best_border = int(color)
        if best_border is None:
            continue

        infos.append(
            {
                "pts": pts,
                "fill": int(mapping[best_border]),
                "seed": seeds[best_border],
            }
        )
    return infos


def _solve_scene_and_collect(g: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int, np.ndarray]]]:
    """
    Solve the full input scene, then identify matched sprites for packing.

    Returns:
      - solved full scene
      - background color
      - list of matched pieces as (r0, c0, cropped_piece)
    """
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "_solve_scene_and_collect: palette mapping must be non-empty"

    matched = _matched_hole_infos(g, bg, mapping, pal_mask)
    out = g.copy()
    for info in matched:
        _paint_striped_hole(out, info["pts"], bg, int(info["fill"]))

    nonbg_mask = (out != bg) & (~pal_mask)
    comps = components_mask(nonbg_mask, connectivity=4)
    comp_id = -np.ones(out.shape, dtype=int)
    comp_boxes: list[tuple[int, int, int, int]] = []
    for idx, pts in enumerate(comps):
        comp_boxes.append(_bbox_from_pts(pts))
        for r, c in pts:
            comp_id[int(r), int(c)] = int(idx)

    piece_ids: set[int] = set()
    for info in matched:
        sr, sc = info["seed"]
        idx = int(comp_id[int(sr), int(sc)])
        if idx >= 0:
            piece_ids.add(idx)

    pieces: list[tuple[int, int, np.ndarray]] = []
    for idx in sorted(piece_ids, key=lambda k: (comp_boxes[k][0], comp_boxes[k][1])):
        r0, c0, r1, c1 = comp_boxes[idx]
        pieces.append((int(r0), int(c0), out[r0 : r1 + 1, c0 : c1 + 1].copy()))

    return out, int(bg), pieces


def _pack_pieces(pieces: list[np.ndarray], bg: int) -> np.ndarray:
    """Pack cropped pieces left-to-right with one background column between them."""
    assert pieces, "_pack_pieces requires at least one piece"
    out_h = max(int(piece.shape[0]) for piece in pieces)
    out_w = sum(int(piece.shape[1]) for piece in pieces) + (len(pieces) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    c0 = 0
    for i, piece in enumerate(pieces):
        h, w = map(int, piece.shape)
        out[0:h, c0 : c0 + w] = piece
        c0 += w
        if i + 1 < len(pieces):
            c0 += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill matched hollow sprites with palette-guided stripes, then output only those sprites as cropped packed pieces.

    Input structure:
    - Background is the dominant outer-border color, with an isolated 2xN or Nx2 palette legend on one grid edge.
    - The legend's outer line lists sprite border colors and the inner line lists their corresponding fill colors.
    - Several hollow single-color sprites sit elsewhere on the canvas; some border colors appear in the palette and at least one distractor sprite does not.

    Rule:
    - Detect the edge palette and read it as a border-color to fill-color mapping.
    - Find enclosed background holes that do not touch the outer grid boundary. If a hole is surrounded mainly by a border color present in the palette, fill that hole using alternating stripes of the mapped color and background. Wider-or-square holes stripe by column; taller holes stripe by row.
    - After all matched holes are striped, discard the original scene layout. Keep only the matched sprites, crop each one to its tight bounding box, and arrange the cropped sprites left-to-right in their original reading order with one background column between neighbors.

    Key insight:
    - The palette still decides which color belongs inside each matched sprite, but the hole geometry now decides whether that color alternates horizontally or vertically. The final answer is not the solved full canvas; it is a repacked strip containing only the sprites that actually used the palette.
    """
    g = np.asarray(grid, dtype=int)
    _solved, bg, pieces_with_pos = _solve_scene_and_collect(g)
    pieces = [piece for _r0, _c0, piece in pieces_with_pos]
    return _pack_pieces(pieces, bg)


# ---------------------------- generator helpers ----------------------------


_GRID_H_MIN, _GRID_H_MAX = 6, 13
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
_P_DIAMOND = 0.2

_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return local bg components not touching the sprite boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _is_connected_sprite(sprite: np.ndarray) -> bool:
    return len(components_mask(sprite, connectivity=4)) == 1


def _make_diamond() -> np.ndarray:
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = True
    s[1, 0] = True
    s[1, 2] = True
    s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a connected hollow sprite with at least one enclosed hole and bounded area."""
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
            if _get_enclosed_holes(test) and _is_connected_sprite(test):
                sprite = test

    if not _get_enclosed_holes(sprite):
        return None
    if int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    if not _is_connected_sprite(sprite):
        return None
    return sprite


def _hole_visible_for_stripes(hpts: np.ndarray) -> bool:
    return int(len(hpts)) >= 2


def _supports_axis(mask: np.ndarray, axis: str) -> bool:
    for hpts in _get_enclosed_holes(mask):
        if _hole_visible_for_stripes(hpts) and _stripe_axis_for_hole(hpts) == axis:
            return True
    return False


def _random_shape(
    rng: random.Random, max_sh: int, max_sw: int, allow_diamond: bool = True
) -> np.ndarray | None:
    sh = rng.randint(3, max_sh)
    sw = rng.randint(3, max_sw)
    if sh + sw > _SPRITE_DIM_SUM_CAP:
        sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
    if sh == 3 and sw == 3 and allow_diamond and rng.random() < _P_DIAMOND:
        mask = _make_diamond()
        return mask if _is_connected_sprite(mask) else None
    return _make_hollow_sprite(sh, sw, rng)


def _make_target_axes(rng: random.Random, num_filled: int) -> list[str]:
    if num_filled == 1:
        return [rng.choice(["horizontal", "vertical"])]
    axes = ["horizontal", "vertical"]
    while len(axes) < num_filled:
        axes.append(rng.choice(["horizontal", "vertical"]))
    rng.shuffle(axes)
    return axes


def _choose_colors(
    rng: random.Random, n_pairs: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick palette colors, one distractor border color, and filled-sprite keys."""
    pool = list(range(1, 10))
    rng.shuffle(pool)
    border_keys = pool[:n_pairs]

    fill_pool = [c for c in range(1, 10) if c not in border_keys]
    rng.shuffle(fill_pool)
    fill_values = fill_pool[:n_pairs]

    distractor_pool = [c for c in range(1, 10) if c not in border_keys and c not in fill_values]
    distractor_color = rng.choice(distractor_pool)

    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    n_pairs = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_pairs), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n_pairs, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> None:
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        pr0, pc0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        pr0, pc0 = rng.randint(0, h - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, h - pal_h), w - pal_w

    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[max(0, pr0 - 1) : min(h, pr0 + pal_h + 1), max(0, pc0 - 1) : min(w, pc0 + pal_w + 1)] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_local_striped_hole(
    grid: np.ndarray, r0: int, c0: int, hpts: np.ndarray, bg: int, fill: int
) -> None:
    hr0, hc0, _, _ = _bbox_from_pts(hpts)
    axis = _stripe_axis_for_hole(hpts)
    for hr, hc in hpts:
        rr = r0 + int(hr)
        cc = c0 + int(hc)
        if axis == "horizontal":
            grid[rr, cc] = int(fill) if (int(hr) - hr0) % 2 == 0 else int(bg)
        else:
            grid[rr, cc] = int(fill) if (int(hc) - hc0) % 2 == 0 else int(bg)


def _paint_local_solid_hole(
    grid: np.ndarray, r0: int, c0: int, hpts: np.ndarray, fill: int
) -> None:
    for hr, hc in hpts:
        grid[r0 + int(hr), c0 + int(hc)] = int(fill)


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
    bg: int,
    striped: bool,
) -> None:
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
    if fill is None:
        return
    for hpts in holes:
        if striped:
            _paint_local_striped_hole(grid, r0, c0, hpts, bg, int(fill))
        else:
            _paint_local_solid_hole(grid, r0, c0, hpts, int(fill))


def _packed_output_from_placed(
    scene: np.ndarray, bg: int, placed: list[dict[str, object]]
) -> np.ndarray:
    pieces: list[tuple[int, int, np.ndarray]] = []
    for info in placed:
        if info["fill"] is None:
            continue
        r0 = int(info["r0"])
        c0 = int(info["c0"])
        mask = info["mask"]
        sh, sw = mask.shape
        pieces.append((r0, c0, scene[r0 : r0 + sh, c0 : c0 + sw].copy()))
    pieces.sort(key=lambda item: (item[0], item[1]))
    return _pack_pieces([piece for _r0, _c0, piece in pieces], bg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))
    total_sprites = num_filled + 1

    target_axes = _make_target_axes(rng, num_filled)
    shapes: list[np.ndarray] = []
    for axis in target_axes:
        mask = None
        for _ in range(100):
            cand = _random_shape(rng, max_sh, max_sw, allow_diamond=False)
            if cand is None:
                continue
            if _supports_axis(cand, axis):
                mask = cand
                break
        if mask is None:
            return None
        shapes.append(mask)

    while len(shapes) < total_sprites:
        cand = _random_shape(rng, max_sh, max_sw, allow_diamond=True)
        if cand is None:
            return None
        shapes.append(cand)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n_pairs, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    inp = np.full((h, w), bg, dtype=int)
    solved_scene = np.full((h, w), bg, dtype=int)
    solid_scene = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(inp, occ, pal, side, rng)
    solved_scene[:] = inp
    solid_scene[:] = inp

    infos: list[tuple[np.ndarray, int, int | None]] = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            color = filled_keys[i]
            fill = mapping[color]
        else:
            color = distractor_color
            fill = None
        infos.append((mask, color, fill))

    placed: list[dict[str, object]] = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)

        solved_scene[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
        solid_scene[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
        inp[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)

        if fill is not None:
            for hpts in holes:
                _paint_local_striped_hole(solved_scene, r0, c0, hpts, bg, int(fill))
                _paint_local_solid_hole(solid_scene, r0, c0, hpts, int(fill))

        placed.append({"r0": r0, "c0": c0, "mask": mask, "fill": fill})
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    for info in placed:
        if info["fill"] is None:
            continue
        r0 = int(info["r0"])
        c0 = int(info["c0"])
        mask = info["mask"]
        for hpts in _get_enclosed_holes(mask):
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    output = _packed_output_from_placed(solved_scene, bg, placed)
    solid_output = _packed_output_from_placed(solid_scene, bg, placed)

    if np.array_equal(output, solid_output):
        return None
    if output.shape == inp.shape and np.array_equal(output, solved_scene):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
