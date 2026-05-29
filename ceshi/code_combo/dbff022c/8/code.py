# Combo variant of dbff022c (index 8)
# combo_spec: D5/1+D7/0
# Name: 棋盘格填充 + 未匹配精灵被删除

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The original task reads a 2xN or Nx2 palette on the outer edge as border-color -> fill-color,
#      then solid-fills every enclosed background hole whose surrounding border color is present.
#    - This combo changes both branches at once:
#      1. matched sprites still use the palette, but their holes are not painted solidly anymore;
#         only cells with even global parity ((r + c) % 2 == 0) receive the mapped fill color.
#      2. unmatched sprites no longer stay unchanged; if a hole is surrounded by a border color that
#         does not appear on the palette key side, the whole sprite component is erased to bg.
#    - Small trace: suppose one hole is surrounded by border color 4 and the palette says 4 -> 6.
#      Original output writes every hole cell as 6. Combo output writes 6 only on even-parity cells
#      and leaves odd-parity cells as 0. If another hollow sprite uses border color 3 and 3 is absent
#      from the palette keys, original output left that sprite untouched; combo output removes all of
#      its border cells, so that sprite disappears entirely.
#
# b) Complexity calibration
#    - The generator keeps exactly one unmatched hollow distractor, so deletion is always visible.
#    - It also requires at least one matched hole with both painted and unpainted checkerboard cells;
#      otherwise the sample would collapse back toward the original solid-fill behavior.
#    - Palette size stays small (2-5 pairs) and sprite count stays 2-4, preserving ARC-scale inference
#      while still forcing the solver to learn two distinct consequences of palette membership.
#
# c) Implicit-assumption audit of the original main()
#    - Original `_fill_enclosed_bg_regions` mutated `out` while also using it for adjacency analysis.
#      That was safe when the only operation was solid fill. It is unsafe once unmatched sprites may be
#      deleted, because analysis order could change later holes. The combo solver therefore analyzes from
#      the original input grid `g` and applies writes afterward.
#    - Original code assumed every enclosed hole with no mapped border could be ignored. That assumption
#      is broken by D7/0; such holes now imply sprite deletion.
#    - Original fill loop assumed every selected hole cell should be overwritten. D5/1 breaks that; the
#      write must be conditional on absolute coordinate parity.
#    - Palette detection assumptions still hold: the palette remains isolated on an edge, the outer line
#      has distinct non-bg keys, and the border background is still dominant.
#
# d) Placement legality
#    - The generator places the palette first and then places each sprite with a 1-cell bg moat from
#      every other object and from the palette. This keeps each enclosed hole attributable to one sprite.
#    - Holes are guaranteed not to touch the outer grid boundary, matching the solver's "interior hole"
#      test.
#    - The unmatched sprite exists only in the input; output is constructed without it, so deletion is
#      not simulated by overlap tricks.
#
# e) Visual observability
#    - Across samples, matched sprites survive but show alternating fill pixels, while one similarly
#      hollow unmatched sprite vanishes altogether. These two outcomes are visually distinct and both are
#      tied to the same observable condition: whether the border color appears on the palette key side.
#    - The checkerboard uses visible grid parity, not hidden color ordering or numeric thresholds.
#
# f) Visual cleanliness
#    - Only three visual layers remain: palette, hollow borders, and checkerboard fills. Deleted sprites
#      leave clean background regions because objects never overlap.
#    - The distractor color is chosen outside the palette key colors and preferably outside fill colors,
#      reducing accidental visual confusion.
#
# g) Change inferability
#    - The palette still provides the fill color directly for surviving sprites.
#    - The extra inference is structural rather than arbitrary: palette-key membership determines survival,
#      and surviving interiors reveal a repeated alternating placement pattern. Because every sample has
#      at least one deleted distractor and one non-solid fill, both parts of the combo are learnable from
#      a small set of examples.


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
                outer_colors = set(map(int, np.unique(seg_outer)))
                inner_colors = set(map(int, np.unique(seg_inner)))
                outer_colors.discard(bg)
                inner_colors.discard(bg)
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                all_colors = outer_colors | inner_colors
                if len(all_colors) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(all_colors), -(c1 - c0), nonbg), "top", c0, c1)
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
                outer_colors = set(map(int, np.unique(seg_outer)))
                inner_colors = set(map(int, np.unique(seg_inner)))
                outer_colors.discard(bg)
                inner_colors.discard(bg)
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                all_colors = outer_colors | inner_colors
                if len(all_colors) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(all_colors), -(c1 - c0), nonbg), "bottom", c0, c1)
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
                outer_colors = set(map(int, np.unique(seg_outer)))
                inner_colors = set(map(int, np.unique(seg_inner)))
                outer_colors.discard(bg)
                inner_colors.discard(bg)
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                all_colors = outer_colors | inner_colors
                if len(all_colors) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(all_colors), -(r1 - r0), nonbg), "left", r0, r1)
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
                outer_colors = set(map(int, np.unique(seg_outer)))
                inner_colors = set(map(int, np.unique(seg_inner)))
                outer_colors.discard(bg)
                inner_colors.discard(bg)
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                all_colors = outer_colors | inner_colors
                if len(all_colors) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(all_colors), -(r1 - r0), nonbg), "right", r0, r1)
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

    for border_color, fill_color in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        border_color = int(border_color)
        if border_color == bg:
            continue
        mapping[border_color] = int(fill_color)

    return mapping, pal_mask


def _checkerboard_fill_region(out: np.ndarray, pts: np.ndarray, fill: int) -> tuple[int, int]:
    """Paint even-parity cells in pts with fill and leave odd-parity cells as bg."""
    painted = 0
    skipped = 0
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        if (rr + cc) % 2 == 0:
            out[rr, cc] = int(fill)
            painted += 1
        else:
            skipped += 1
    return painted, skipped


def _apply_combo_transform(
    out: np.ndarray, g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Analyze holes from the original input grid, then:
    - checkerboard-fill holes whose dominant border color is a palette key
    - delete components whose dominant border color is absent from the palette
    """
    h, w = g.shape
    obj_mask = (g != bg) & (~pal_mask)
    obj_comps = components_mask(obj_mask, connectivity=4)
    comp_id = np.full((h, w), -1, dtype=int)
    for cid, pts in enumerate(obj_comps):
        for r, c in pts:
            comp_id[int(r), int(c)] = int(cid)

    fills: list[tuple[np.ndarray, int]] = []
    delete_ids: set[int] = set()
    bg_mask = (g == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
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
        for color, count in adj_counts.items():
            if color in mapping and count > mapped_count:
                mapped_border = int(color)
                mapped_count = int(count)

        if mapped_border is not None:
            fills.append((pts, int(mapping[mapped_border])))
            continue

        unmatched_border = max(adj_counts.items(), key=lambda item: item[1])[0]
        for cid in adj_color_to_ids.get(int(unmatched_border), set()):
            delete_ids.add(int(cid))

    for pts, fill in fills:
        _checkerboard_fill_region(out, pts, fill)
    for cid in delete_ids:
        for r, c in obj_comps[cid]:
            out[int(r), int(c)] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use an edge palette to both filter hollow sprites and checker-fill the ones that survive.

    Input structure:
    - The outer border is mostly background, with several hollow single-color sprites inside the grid.
    - A compact 2xN or Nx2 legend lies on one edge; its outer line lists sprite border colors and its
      inner line lists the corresponding fill colors.

    Rule:
    - Read the edge legend as a border-to-fill mapping.
    - Find enclosed background regions that act as sprite interiors, ignoring the palette area itself.
    - If a hole is surrounded by a border color found in the palette, fill only the even-parity hole
      cells with the mapped color and leave the odd-parity hole cells as background.
    - If a hole is surrounded by a border color absent from the palette, erase the whole touching sprite
      back to the background color.

    Key insight:
    - The palette does two jobs at once: it says which border colors survive and what fill color they use.
    - Surviving interiors are not solid; the mapped color must be written on alternating grid positions
      based on absolute row-plus-column parity.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _apply_combo_transform(out, g, bg, mapping, pal_mask)
    return out


# --------------- generator (combo variant) ---------------

# Grid
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7
_TINY_H_THRESHOLD = 5

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

# Sprite mutation
_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

# Generation
_MAX_RETRIES = 300


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point-arrays for bg components not touching the sprite boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped hollow mask."""
    sprite = np.zeros((3, 3), dtype=bool)
    sprite[0, 1] = True
    sprite[1, 0] = True
    sprite[1, 2] = True
    sprite[2, 1] = True
    return sprite


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a hollow sprite with at least one enclosed hole and limited border area."""
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
        border_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(border_cells)
        for rr, cc in border_cells[: rng.randint(1, min(4, len(border_cells)))]:
            test = sprite.copy()
            test[rr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test
    holes = _get_enclosed_holes(sprite)
    if not holes or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(
    rng: random.Random, n_pairs: int, num_matched: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick palette colors, one distractor color, and the matched sprite border keys."""
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:n_pairs]
    rng.shuffle(all_colors)
    fill_values = all_colors[:n_pairs]
    distractor_candidates = [
        c for c in range(1, 10) if c not in border_keys and c not in fill_values
    ]
    if not distractor_candidates:
        distractor_candidates = [c for c in range(1, 10) if c not in border_keys]
    distractor_color = int(rng.choice(distractor_candidates))
    mapping = dict(zip(border_keys, fill_values))
    matched_keys = rng.sample(border_keys, num_matched)
    return border_keys, fill_values, distractor_color, mapping, matched_keys


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


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> None:
    """Place the palette on the chosen edge and reserve a 1-cell padding zone."""
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
    occ[max(0, r0 - 1) : min(h, r0 + pal_h + 1), max(0, c0 - 1) : min(w, c0 + pal_w + 1)] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements where the sprite fits, stays separated, and keeps holes internal."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hole in holes
                for hr, hc in hole
            ):
                continue
            positions.append((int(r0), int(c0)))
    return positions


def _paint_sprite_border(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    """Paint only the border cells of a sprite."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)


def _paint_checkerboard_holes(
    grid: np.ndarray, r0: int, c0: int, holes: list[np.ndarray], fill: int
) -> tuple[int, int]:
    """Checker-fill the placed holes and report painted vs skipped cells."""
    painted = 0
    skipped = 0
    for pts in holes:
        global_pts = np.array([(r0 + int(hr), c0 + int(hc)) for hr, hc in pts], dtype=int)
        add_painted, add_skipped = _checkerboard_fill_region(grid, global_pts, int(fill))
        painted += add_painted
        skipped += add_skipped
    return painted, skipped


def _clear_holes(grid: np.ndarray, r0: int, c0: int, holes: list[np.ndarray], bg: int) -> None:
    """Reset the placed holes to background."""
    for pts in holes:
        for hr, hc in pts:
            grid[r0 + int(hr), c0 + int(hc)] = int(bg)


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

    num_matched = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))

    matched_shapes: list[np.ndarray] = []
    for _ in range(num_matched):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        matched_shapes.append(mask)

    sh = rng.randint(3, max_sh)
    sw = rng.randint(3, max_sw)
    if sh + sw > _SPRITE_DIM_SUM_CAP:
        sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
    if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
        distractor_shape = _make_diamond()
    else:
        distractor_shape = _make_hollow_sprite(sh, sw, rng)
    if distractor_shape is None:
        return None

    border_keys, fill_values, distractor_color, mapping, matched_keys = _choose_colors(
        rng, n_pairs, num_matched
    )
    pal = _build_palette(side, border_keys, fill_values)

    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, pal, side, rng)

    placed_matched: list[tuple[int, int, list[np.ndarray]]] = []
    total_checker_cells = 0
    total_bg_kept = 0
    for mask, border_color in zip(matched_shapes, matched_keys, strict=False):
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite_border(output, r0, c0, mask, int(border_color))
        painted, skipped = _paint_checkerboard_holes(output, r0, c0, holes, int(mapping[int(border_color)]))
        placed_matched.append((int(r0), int(c0), holes))
        total_checker_cells += painted
        total_bg_kept += skipped
        sh2, sw2 = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh2 + 1), max(0, c0 - 1) : min(w, c0 + sw2 + 1)] = True

    if total_checker_cells == 0 or total_bg_kept == 0:
        return None

    inp = output.copy()
    for r0, c0, holes in placed_matched:
        _clear_holes(inp, int(r0), int(c0), holes, bg)

    distractor_holes = _get_enclosed_holes(distractor_shape)
    positions = _find_valid_positions(h, w, occ, distractor_shape, distractor_holes)
    if not positions:
        return None
    r0, c0 = rng.choice(positions)
    _paint_sprite_border(inp, r0, c0, distractor_shape, distractor_color)

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
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
