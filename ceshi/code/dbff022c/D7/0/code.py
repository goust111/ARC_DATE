# Variant of dbff022c
# Dimension: D7 — 未在调色板中出现的精灵的处理方式 (transform)
# Change: 边框颜色不在调色板中的精灵保持原样不做任何修改... → 未匹配精灵被删除（用背景色覆盖）

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the D7 variant
# a) Concrete behavior:
#    - Input still contains a 2xN or Nx2 edge palette plus several hollow monochrome sprites.
#    - Palette outer row/col lists border colors; inner row/col lists fill colors.
#    - Sprites whose border color is a palette key survive and have their enclosed bg holes filled.
#    - A distractor hollow sprite whose border color is absent from the palette is present in the input,
#      but absent in the output: its border is erased back to bg, so the whole sprite disappears.
#    - Small trace example: if a hole is surrounded by color 7 and 7 is in the palette, fill that hole.
#      If a hole is surrounded by color 3 and 3 is not in the palette, erase every sprite component
#      touching that hole instead of leaving it unchanged as in the original task.
# b) Complexity calibration:
#    - We keep 2-4 sprites and exactly one unmatched hollow distractor, so the new branch is always
#      visible without adding many new object types.
#    - Generator still allows irregular hollow shapes and multi-hole sprites, so deletion is not a trivial
#      bbox operation; the solver must reason from enclosed holes back to the owning sprite.
# c) Old implicit assumptions to audit:
#    - Original `_fill_enclosed_bg_regions` assumed every relevant hole either maps to a palette key or can
#      be safely ignored. That is no longer valid: "ignored" would regress to the old behavior.
#    - Filling holes in-place was harmless before. Now destructive deletion is possible, so the solver first
#      analyzes holes/components from the original input and only then mutates the output, avoiding order bugs.
#    - Original logic only needed the adjacent color. The variant also needs the adjacent sprite component ids
#      so an unmatched sprite can be erased completely rather than just leaving its hole bg.
# d) Placement legality:
#    - Palette remains isolated on one edge.
#    - All sprites are placed with >=1 bg padding from each other and the palette, and all enclosed holes stay
#      away from the grid boundary, so each hole corresponds to exactly one local sprite situation.
#    - The distractor is placed only in the input grid; the output omits it entirely.
# e) Visual observability:
#    - Each sample contains both surviving matched sprites and one disappearing unmatched sprite.
#    - This makes the palette read as both a color map and a keep/delete filter, which is visible from a
#      few examples because one hollow object vanishes while similarly structured keyed objects are filled.
# f) Visual cleanliness:
#    - No overlaps, no edge-adjacent palette ambiguity, and only one deletion branch per sample.
#    - Objects remain separated by background, so removed sprites leave clean empty regions.
# g) Color inference:
#    - Distractor color is never a palette key and prefers not to appear in the palette fill line either.
#    - The relevant observable condition is "border color appears on the palette key side", not any hidden
#      ordering or numeric color-id property.


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
        outside the palette region; this prevents accidentally absorbing parts of large sprites
        that touch the grid edge.
        """
        h2, w2 = g.shape
        pal = np.zeros((h2, w2), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h2 - 2 : h2, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w2 - 2 : w2] = True

        for r, c in zip(*np.where(pal), strict=False):
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h2 and 0 <= cc < w2 and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[0, :]
        b = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[h - 1, :]
        b = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, 0]
        b = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, w - 1]
        b = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(r1 - r0), nonbg)
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


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Returns (mapping border->fill, palette_mask).
    """
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
    else:  # right
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b == bg:
            continue
        mapping[b] = f

    return mapping, pal_mask


def _fill_or_delete_sprites(out: np.ndarray, g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Process enclosed bg holes from the original grid.

    - If the dominant adjacent border color is in the palette mapping, fill the hole.
    - Otherwise, erase every sprite component of that dominant color touching the hole.
    """
    h, w = g.shape
    obj_mask = (g != bg) & (~pal_mask)
    obj_comps = components_mask(obj_mask, connectivity=4)
    comp_id = np.full((h, w), -1, dtype=int)
    for cid, pts in enumerate(obj_comps):
        for r, c in pts:
            comp_id[int(r), int(c)] = int(cid)

    bg_mask = (g == bg) & (~pal_mask)
    hole_comps = components_mask(bg_mask, connectivity=4)
    delete_ids: set[int] = set()

    for pts in hole_comps:
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

        border = max(adj_counts.items(), key=lambda item: item[1])[0]
        for cid in adj_color_to_ids.get(int(border), set()):
            delete_ids.add(int(cid))

    for cid in delete_ids:
        for r, c in obj_comps[int(cid)]:
            out[int(r), int(c)] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use an edge palette to decide which hollow sprites survive and how their interiors are colored.

    Input structure:
    - bg is the dominant border color. Several hollow single-color sprites sit away from the palette.
    - A compact 2×N or N×2 legend on one grid edge pairs sprite border colors with fill colors.

    Rule:
    - Read the edge legend as a border→fill mapping.
    - Find enclosed bg regions that act as sprite holes, ignoring the palette area itself.
    - If a hole's surrounding border color appears in the palette, fill that hole with the mapped color.
    - If that surrounding border color is absent from the palette, delete the touching sprite by repainting it with bg.

    Key insight: The palette does double duty: it gives the fill color for matching borders and also marks which border colors are allowed to remain in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_or_delete_sprites(out, g, bg, mapping, pal_mask)

    return out


# --------------- generator (variant) ---------------

# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7
_TINY_H_THRESHOLD = 5

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 300
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return list of point-arrays for bg components not touching sprite boundary."""
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
    """Create hollow sprite (h x w) with at least 1 enclosed bg hole, cell count <= _SPRITE_CELL_MAX."""
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


def _choose_colors(rng: random.Random, n_pairs: int, num_filled: int) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick palette border/fill colors, distractor color, and filled-sprite keys."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n_pairs]
    rng.shuffle(all_c)
    fill_values = all_c[:n_pairs]
    distractor_cands = [c for c in range(1, 10) if c not in border_keys and c not in fill_values]
    if not distractor_cands:
        distractor_cands = [c for c in range(1, 10) if c not in border_keys]
    distractor_color = rng.choice(distractor_cands)
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, int(distractor_color), mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the given edge side."""
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


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place palette on the chosen edge and mark its occupancy (with 1-cell padding)."""
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


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]) -> list[tuple[int, int]]:
    """Find all (r0, c0) where mask fits without overlap and holes don't touch grid boundary."""
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
            positions.append((int(r0), int(c0)))
    return positions


def _paint_sprite(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int, fill: int | None, holes: list[np.ndarray]) -> None:
    """Render a sprite onto the grid: border cells in color, hole cells in fill (if given)."""
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
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))

    matched_shapes: list[np.ndarray] = []
    for _ in range(num_filled):
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

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n_pairs, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, pal, side, rng)

    placed_filled: list[tuple[int, int, list[np.ndarray]]] = []
    for mask, border_color in zip(matched_shapes, filled_keys, strict=False):
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(output, r0, c0, mask, int(border_color), int(mapping[int(border_color)]), holes)
        placed_filled.append((int(r0), int(c0), holes))
        sh2, sw2 = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh2 + 1), max(0, c0 - 1) : min(w, c0 + sw2 + 1)] = True

    inp = output.copy()
    for r0, c0, holes in placed_filled:
        _clear_holes(inp, int(r0), int(c0), holes, bg)

    distractor_holes = _get_enclosed_holes(distractor_shape)
    positions = _find_valid_positions(h, w, occ, distractor_shape, distractor_holes)
    if not positions:
        return None
    r0, c0 = rng.choice(positions)
    _paint_sprite(inp, r0, c0, distractor_shape, distractor_color, None, distractor_holes)

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
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
        r = _gen_attempt()
        if r is not None:
            return r
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
