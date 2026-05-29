# Combo variant of dbff022c (index 32)
# combo_spec: D6/0+D7/0
# Name: 裁剪调色板 + 删除未匹配精灵

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Input still contains one isolated 2xN / Nx2 palette on a grid edge and several hollow monochrome sprites.
#    - Relative to the original task, two steps change together:
#      1. A hole surrounded by a border color that appears on the palette key side is filled with the mapped color.
#      2. A hole surrounded by a border color absent from the palette no longer stays unchanged; the whole touching
#         sprite component is erased to bg.
#      3. After those content edits are finished on the full input canvas, the full 2-cell edge strip carrying the
#         palette is cropped away from the output.
#    - Small trace: if the palette is on the top strip, original output would keep rows 0..1 and leave an unmatched
#      sprite visible. The combo output first fills matched holes, deletes the unmatched hollow sprite, then returns
#      only rows 2..H-1. The visible delta is both spatial (shape shrinks by 2 on one axis) and semantic (one sprite
#      disappears).
# b) Complexity calibration:
#    - Every sample contains at least one matched sprite and exactly one unmatched hollow distractor, so both branches
#      are observable without introducing many object types.
#    - Generator builds the final output on the cropped canvas first, then embeds it back into the full input canvas.
#      This prevents accidental placement inside the removable strip and keeps the crop effect obvious.
# c) Implicit assumptions audit:
#    - `_bg_boundary` still assumes the true outer border is mostly bg. Generator keeps all sprites at least one cell
#      away from the cropped canvas border, so after embedding, only bg and palette occupy the full-grid boundary.
#    - `_best_palette` assumes the only meaningful edge object is the palette. Generator reserves the removable strip
#      for the palette alone and keeps sprites away from it.
#    - Hole analysis now drives both filling and deletion, so mutation order matters. Solver analyzes holes and object
#      components from the original input `g`, not from a partially edited grid, avoiding fill/delete interference.
#    - Cropping assumes the palette always lies inside the full 2-row/2-col edge strip of its side. Generator places
#      it only there.
# d) Placement legality:
#    - Sprites are placed on the cropped canvas with a 1-cell bg moat from each other and from the cropped border.
#    - The full input is formed by embedding that cropped canvas into the non-removed region and placing the palette
#      inside the removable strip, so no object overlaps or out-of-bounds placements can occur.
# e) Visual observability:
#    - Across 2-4 examples, a human can see that the palette is input-only because the output is smaller and the
#      palette disappears entirely.
#    - The same examples also show that one hollow sprite survives and fills while another hollow sprite vanishes,
#      making the palette act as both a color legend and a keep/delete filter.
# f) Visual cleanliness:
#    - Palette, sprites, and empty background occupy disjoint regions with at least one bg cell of separation.
#    - Deleted distractors leave clean empty gaps in the retained field rather than overlapping with the crop area.
# g) Inference guardrails:
#    - The unmatched distractor color is never a palette key and prefers not to appear as a fill color either.
#    - We keep the color vocabulary modest and repeat the palette-key mechanism via at least one matched sprite per
#      sample, so the delete condition is structurally inferable from visible border-to-palette correspondence.


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
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        """
        Palette cells (excluding bg) should not be 4-neigh adjacent to same-colored cells
        outside the palette region; this prevents accidentally absorbing parts of sprites.
        """
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
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
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
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(border_keys)), len(union), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
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
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(border_keys)), len(union), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
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
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(border_keys)), len(union), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
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
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(border_keys)), len(union), -(r1 - r0), nonbg)
                cand = (score, "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray, tuple[str, int, int]]:
    """Returns (mapping border->fill, palette_mask, palette_info)."""
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
        b = int(b)
        f = int(f)
        if b == bg:
            continue
        mapping[b] = f

    return mapping, pal_mask, (side, start, end)


def _fill_or_delete_sprites(
    out: np.ndarray, g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Analyze enclosed bg holes from the original grid.

    - If a hole's dominant border color appears in the palette mapping, fill that hole.
    - Otherwise erase the touching sprite component(s) of the dominant border color.
    """
    h, w = g.shape
    obj_mask = (g != bg) & (~pal_mask)
    obj_comps = components_mask(obj_mask, connectivity=4)
    comp_id = np.full((h, w), -1, dtype=int)
    for cid, pts in enumerate(obj_comps):
        for r, c in pts:
            comp_id[int(r), int(c)] = int(cid)

    bg_mask = (g == bg) & (~pal_mask)
    delete_ids: set[int] = set()

    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        adj_color_to_ids: dict[int, set[int]] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if pal_mask[rr, cc]:
                    continue
                v = int(g[rr, cc])
                if v == bg:
                    continue
                adj_counts[v] = adj_counts.get(v, 0) + 1
                cid = int(comp_id[rr, cc])
                if cid >= 0:
                    adj_color_to_ids.setdefault(v, set()).add(cid)

        if not adj_counts:
            continue

        mapped_border = None
        mapped_count = -1
        for col, cnt in adj_counts.items():
            if col in mapping and cnt > mapped_count:
                mapped_count = cnt
                mapped_border = int(col)
        if mapped_border is not None:
            fill = int(mapping[mapped_border])
            for r, c in pts:
                out[int(r), int(c)] = fill
            continue

        border = int(max(adj_counts.items(), key=lambda item: item[1])[0])
        for cid in adj_color_to_ids.get(border, set()):
            delete_ids.add(int(cid))

    for cid in delete_ids:
        for r, c in obj_comps[cid]:
            out[int(r), int(c)] = bg


def _crop_palette_strip(out: np.ndarray, side: str) -> np.ndarray:
    """Remove the full 2-cell edge strip that contains the palette legend."""
    if side == "top":
        return out[2:, :].copy()
    if side == "bottom":
        return out[:-2, :].copy()
    if side == "left":
        return out[:, 2:].copy()
    return out[:, :-2].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use an edge palette to keep/fill matched hollow sprites, delete unmatched ones, then crop away the palette strip.

    Input structure:
    - The full grid has a dominant background border and one isolated 2xN or Nx2 palette sitting on a single edge.
    - Away from that edge are hollow single-color sprites; some border colors appear on the palette key side and one
      hollow distractor border color does not.

    Rule:
    - Detect the edge palette and read it as a border-color to fill-color mapping.
    - For each enclosed background hole outside the palette, look at its surrounding border color on the original grid.
      If that border color is a palette key, fill the hole with the mapped color; otherwise erase the touching sprite
      component back to background.
    - After all fills and deletions are resolved on the full grid, crop off the entire 2-cell strip that carried the
      palette.

    Key insight: The palette is pure control information. It both decides which border colors survive and what color
    their interiors become, and then the whole palette-bearing side disappears from the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask, pal_info = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_or_delete_sprites(out, g, bg, mapping, pal_mask)

    side, _start, _end = pal_info
    return _crop_palette_strip(out, side)


# --------------- generator (combo variant) ---------------

# Full input dimensions. One axis loses exactly 2 cells after cropping.
_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 12, 15

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

# Sprite mutation probabilities
_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

# Generation
_MAX_RETRIES = 500


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components that stay fully inside the sprite bbox."""
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
    """Create a hollow sprite with at least one enclosed hole."""
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

    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(
    rng: random.Random, n_pairs: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick palette pairs, an unmatched border color, and which keys appear on sprites."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n_pairs]
    rng.shuffle(all_c)
    fill_values = all_c[:n_pairs]
    distractor_cands = [c for c in range(1, 10) if c not in border_keys and c not in fill_values]
    if not distractor_cands:
        distractor_cands = [c for c in range(1, 10) if c not in border_keys]
    distractor_color = int(rng.choice(distractor_cands))
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the chosen edge."""
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


def _place_palette(grid: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place the palette somewhere inside the removable 2-cell edge strip."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        r0, c0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        r0, c0 = rng.randint(0, h - pal_h), 0
    else:
        r0, c0 = rng.randint(0, h - pal_h), w - pal_w
    grid[r0 : r0 + pal_h, c0 : c0 + pal_w] = pal


def _cropped_shape(side: str, h: int, w: int) -> tuple[int, int]:
    """Shape after removing the palette strip."""
    if side in ("top", "bottom"):
        return h - 2, w
    return h, w - 2


def _embed_cropped(full: np.ndarray, crop: np.ndarray, side: str) -> None:
    """Paste the cropped field back into the full canvas."""
    if side == "top":
        full[2:, :] = crop
    elif side == "bottom":
        full[:-2, :] = crop
    elif side == "left":
        full[:, 2:] = crop
    else:
        full[:, :-2] = crop


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    """Find placements with a 1-cell moat from the cropped border and other sprites."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            positions.append((int(r0), int(c0)))
    return positions


def _paint_sprite(
    grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int, fill: int | None, holes: list[np.ndarray]
) -> None:
    """Render a sprite onto a cropped canvas."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + int(hr), c0 + int(hc)] = int(fill)


def _clear_holes(grid: np.ndarray, r0: int, c0: int, holes: list[np.ndarray], bg: int) -> None:
    """Set all enclosed hole cells for a placed sprite back to bg."""
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + int(hr), c0 + int(hc)] = int(bg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    crop_h, crop_w = _cropped_shape(side, h, w)

    if crop_h < 5 or crop_w < 8:
        return None

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, crop_h - 2)
    max_sw = min(_SPRITE_W_MAX, crop_w - 2)
    if max_sh < 3 or max_sw < 3:
        return None

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))
    total_sprites = num_filled + 1

    shapes: list[np.ndarray] = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sw > max_sw:
            sw = max_sw
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n_pairs, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    crop_out = np.full((crop_h, crop_w), bg, dtype=int)
    crop_in = np.full((crop_h, crop_w), bg, dtype=int)
    occ = np.zeros((crop_h, crop_w), dtype=bool)

    matched_infos: list[tuple[np.ndarray, int, int]] = []
    for i in range(num_filled):
        key = int(filled_keys[i])
        matched_infos.append((shapes[i], key, int(mapping[key])))
    distractor_shape = shapes[-1]

    for mask, color, fill in matched_infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(crop_h, crop_w, occ, mask)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(crop_out, r0, c0, mask, color, fill, holes)
        _paint_sprite(crop_in, r0, c0, mask, color, fill, holes)
        _clear_holes(crop_in, r0, c0, holes, bg)
        sh2, sw2 = mask.shape
        occ[max(0, r0 - 1) : min(crop_h, r0 + sh2 + 1), max(0, c0 - 1) : min(crop_w, c0 + sw2 + 1)] = True

    distractor_holes = _get_enclosed_holes(distractor_shape)
    positions = _find_valid_positions(crop_h, crop_w, occ, distractor_shape)
    if not positions:
        return None
    r0, c0 = rng.choice(positions)
    _paint_sprite(crop_in, r0, c0, distractor_shape, distractor_color, None, distractor_holes)

    full_in = np.full((h, w), bg, dtype=int)
    _embed_cropped(full_in, crop_in, side)
    _place_palette(full_in, pal, side, rng)

    border = np.concatenate([full_in[0], full_in[-1], full_in[1:-1, 0], full_in[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(full_in), crop_out):
            return full_in
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
