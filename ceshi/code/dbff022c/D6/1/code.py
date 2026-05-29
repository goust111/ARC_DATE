# Variant of dbff022c
# Dimension: D6 — 网格输出形状 (grid_shape)
# Change: 输出网格与输入网格完全相同大小，调色板区域保持不变，仅精灵内部被修改... → 紧凑裁剪为所有精灵的最小外接矩形

# Design analysis for the D6 variant
# a) Concrete behavior and trace
#    - Input still contains two kinds of structures: one edge palette (2xN / Nx2) and several hollow sprites.
#    - Original output kept the full canvas: palette stayed on the edge, sprites stayed in place, only mapped holes changed
#      from bg to their fill colors.
#    - New output discards the whole outer canvas and the palette. After filling the eligible holes, we crop to the minimal
#      rectangle containing every sprite cell. Example: if a top palette occupies rows 0-1 and two sprites live in rows 4-8,
#      original logic returned the full HxW solved grid; the variant returns only rows/cols spanning those sprites.
#    - The rule change therefore lives after filling, at the "what region is returned" stage, not in palette reading or hole filling.
# b) Complexity calibration
#    - We keep 2-4 sprites, 2-5 palette entries, and one distractor sprite. This preserves the original ambiguity pressure:
#      the solver must still infer the palette and decide which holes are fillable.
#    - To make D6 visible, generation now guarantees that sprites do not touch the full-grid border. This prevents the crop from
#      silently degenerating to the original canvas size.
# c) Implicit-assumption review of the original main()
#    - _bg_boundary() assumes the outer border is mostly bg. This still holds because the palette remains small and sprites are
#      generated away from the grid boundary.
#    - _best_palette() assumes a single isolated edge palette whose outer stripe provides unique border keys. Unchanged.
#    - _fill_enclosed_bg_regions() treats a bg component as a hole iff it does not touch the full-grid boundary. This remains valid
#      because all sprite holes stay strictly internal to the full grid.
#    - The old "return out" relied on output size being fixed. That assumption is exactly what D6 breaks, so the new main() must
#      crop after filling and must exclude the palette from the crop mask.
# d) Placement legality
#    - Sprites still cannot overlap each other or the palette, and still keep a 1-cell bg moat between objects.
#    - New generator constraint: the entire sprite bbox must lie at least 1 cell inside the full grid boundary. This guarantees a
#      proper crop margin while preserving solver assumptions about enclosed holes.
# e) Visual observability
#    - From examples, a human can see that the palette disappears in the output and the remaining picture is a tight window around
#      the sprites. The crop boundary is therefore a visible, structural change rather than a hidden metadata change.
#    - The crop is determined by spatial extent only, never by color ids or sorting by numeric color values.
# f) Visual cleanliness
#    - We avoid border-touching sprites, so the cropped output is a compact cluster of sprites without unrelated empty frame area.
#    - The palette stays isolated in the input, which keeps the "legend" visually separate from the sprite field.
# g) Color-change inference
#    - Color semantics stay original: palette outer stripe = border keys, inner stripe = fill colors. The crop does not introduce a
#      second color rule, so the task remains inferable from a few examples.

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, neighbors4


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

    def consider_top():
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
                if len(border_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom():
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
                if len(border_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left():
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
                if len(border_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right():
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
                if len(border_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
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

    for bcol, fcol in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        bcol = int(bcol)
        fcol = int(fcol)
        if bcol == bg:
            continue
        mapping[bcol] = fcol

    return mapping, pal_mask


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Find background regions (excluding the palette area) that do NOT touch the grid boundary,
    and fill each region based on the most common adjacent border color (mapped via palette).
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
                if 0 <= rr < h and 0 <= cc < w:
                    if pal_mask[rr, cc]:
                        continue
                    v = int(out[rr, cc])
                    if v == bg:
                        continue
                    adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border_col = None
        best = -1
        for col, cnt in adj_counts.items():
            if col not in mapping:
                continue
            if cnt > best:
                best = cnt
                border_col = col
        if border_col is None:
            continue

        fill = int(mapping[border_col])
        for r, c in pts:
            out[r, c] = fill


def _tight_crop_sprites(grid: np.ndarray, pal_mask: np.ndarray, bg: int) -> np.ndarray:
    """Crop to the bbox of all non-bg sprite cells, explicitly excluding the palette."""
    sprite_mask = (grid != bg) & (~pal_mask)
    bb = bbox(sprite_mask)
    assert bb is not None, "_tight_crop_sprites: expected at least one sprite pixel"
    return crop_bbox(grid, bb)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites using an edge legend, then return only the sprite field.

    Input structure:
    - bg is the dominant border color of the full canvas.
    - One isolated 2×N or N×2 palette sits on a grid edge; its outer stripe lists sprite-border colors and its inner stripe lists the matching fill colors.
    - Away from that palette sit several hollow sprites, including at least one palette-matching sprite and one distractor whose border color is absent from the palette.

    Rule:
    - Read the edge palette as a border→fill lookup table.
    - Fill each enclosed bg hole whose surrounding border color appears in that lookup; leave unmapped holes unchanged.
    - After filling, discard the palette and all surrounding empty frame area by tightly cropping to the minimal rectangle containing every sprite.

    Key insight: The palette still determines which hollow regions get colored, but the final answer is no longer the solved full canvas — it is the tight sprite-only crop inferred from the non-palette foreground extent.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return _tight_crop_sprites(out, pal_mask, bg)


# --------------- generator ---------------

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 11, 15
_SMALL_H_THRESHOLD = 7
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

# Sprite mutation probabilities
_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

# Generation
_MAX_RETRIES = 300


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point-arrays for bg components not touching the sprite bbox boundary."""
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
    """Create a hollow sprite with at least one enclosed hole and bounded cell count."""
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
    rng: random.Random, n_entries: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick palette colors and the distractor border color."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n_entries]
    rng.shuffle(all_c)
    fill_values = all_c[:n_entries]
    distractor_color = rng.choice([c for c in range(1, 10) if c not in border_keys])
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette for the chosen edge side."""
    n_entries = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_entries), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n_entries, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> np.ndarray:
    """Place the palette on the requested edge and return its exact mask."""
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

    pal_mask = np.zeros((h, w), dtype=bool)
    pal_mask[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = True
    return pal_mask


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """
    Find all placements where the sprite stays 1 cell inside the full-grid border,
    does not overlap occupied space, and keeps its holes away from the full-grid edge.
    """
    sh, sw = mask.shape
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if r0 == 0 or c0 == 0 or r0 + sh == h or c0 + sw == w:
                continue
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


def _paint_sprite(
    grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int, fill: int | None, holes: list[np.ndarray]
) -> None:
    """Render a sprite onto the grid: border cells in color, hole cells in fill if provided."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n_entries = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, h - 2)
    max_sw = min(_SPRITE_W_MAX, w - 2)
    if h <= _TINY_H_THRESHOLD:
        max_sh = min(max_sh, 3)
    if max_sh < 3 or max_sw < 3:
        return None

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n_entries))
    total_sprites = num_filled + 1

    shapes = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sw > max_sw:
            return None
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n_entries, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    pal_mask = _place_palette(grid, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            color = filled_keys[i]
            fill = mapping[color]
        else:
            color = distractor_color
            fill = None
        infos.append((mask, color, fill))

    placed = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    solved_full = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill in placed:
        if fill is None:
            continue
        for hpts in _get_enclosed_holes(mask):
            for hr, hc in hpts:
                inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    expected = _tight_crop_sprites(solved_full, pal_mask, bg)
    if not (expected.shape[0] < inp.shape[0] and expected.shape[1] < inp.shape[1]):
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
