import numpy as np

from arc_utils import components_mask, neighbors4
import random


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
                # Border keys should be unique (excluding bg) to form a well-defined mapping.
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
    """
    Returns (mapping border->fill, palette_mask).
    """
    h, w = g.shape
    info = _best_palette(g, bg)
    side, start, end = info
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

        # Prefer an adjacent color that exists in the mapping; pick the most frequent border.
        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col not in mapping:
                continue
            if cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        fill = int(mapping[border])
        for r, c in pts:
            out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites' interiors using a color-palette legend on the grid edge.

    Input structure:
    - bg = most common color. Several hollow sprites with distinct border colors.
    - A small 2×N or N×2 palette block sits on one grid edge; one row/column lists border colors, the other lists corresponding fill colors.

    Rule:
    - Locate the palette legend on a grid edge and read it as a border→fill color mapping.
    - Find enclosed bg regions (holes) inside sprites that don't touch the grid boundary.
    - For each hole, identify the most common adjacent border color, look it up in the palette, and fill the hole with the mapped fill color.

    Key insight: The palette's edge position and orientation determine reading direction; border colors in the palette uniquely key into fill colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)

    return out

# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6        # below this, prefer left/right palette
_SMALL_H_PALETTE_FLIP = 0.7   # probability of flipping top/bottom -> left/right
_TINY_H_THRESHOLD = 5         # below this, cap sprite height at 3

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10      # if sh + sw exceeds this, clamp sw
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

# Sprite mutation probabilities
_P_HORIZ_WALL = 0.3           # add horizontal internal wall
_P_VERT_WALL = 0.3            # add vertical internal wall
_P_BORDER_REMOVAL = 0.4       # remove border cells for irregular shapes
_P_DIAMOND = 0.2              # use diamond shape for 3x3

# Generation
_MAX_RETRIES = 300
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite):
    """Return list of point-arrays for bg components not touching sprite boundary."""
    h, w = sprite.shape
    return [pts for pts in components_mask(~sprite, connectivity=4)
            if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)]


def _make_diamond():
    """Return a 3x3 diamond-shaped boolean mask."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h, w, rng):
    """Create hollow sprite (h x w) with at least 1 enclosed bg hole, cell count <= _SPRITE_CELL_MAX."""
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False
    # Optionally add internal horizontal wall
    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - 1 - n + 1)
        sprite[br, s:s + n] = True
    # Optionally add internal vertical wall
    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - 1 - n + 1)
        sprite[s:s + n, bc] = True
    # Optionally remove border cells for irregularity
    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [(r, c) for r in range(h) for c in range(w)
                  if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))]
        rng.shuffle(bcells)
        for cr, cc in bcells[:rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test
    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(rng, N, num_filled):
    """Pick palette border/fill colors, distractor color, and filled-sprite keys."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:N]
    rng.shuffle(all_c)
    fill_values = all_c[:N]
    dist_cands = [c for c in range(1, 10) if c not in border_keys]
    distractor_color = rng.choice(dist_cands)
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side, border_keys, fill_values):
    """Construct the 2xN or Nx2 palette array for the given edge side."""
    N = len(border_keys)
    if side in ('top', 'bottom'):
        pal = np.zeros((2, N), dtype=int)
        outer, inner = (0, 1) if side == 'top' else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((N, 2), dtype=int)
        outer, inner = (0, 1) if side == 'left' else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid, occ, pal, side, rng):
    """Place palette on the chosen edge and mark its occupancy (with 1-cell padding)."""
    H, W = grid.shape
    pal_h, pal_w = pal.shape
    if side == 'top':
        pr0, pc0 = 0, rng.randint(0, W - pal_w)
    elif side == 'bottom':
        pr0, pc0 = H - pal_h, rng.randint(0, W - pal_w)
    elif side == 'left':
        pr0, pc0 = rng.randint(0, H - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, H - pal_h), W - pal_w
    grid[pr0:pr0 + pal_h, pc0:pc0 + pal_w] = pal
    occ[max(0, pr0 - 1):min(H, pr0 + pal_h + 1),
        max(0, pc0 - 1):min(W, pc0 + pal_w + 1)] = True


def _find_valid_positions(H, W, occ, mask, holes):
    """Find all (r0, c0) where mask fits without overlap and holes don't touch grid boundary."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            if occ[max(0, r0 - 1):min(H, r0 + sh + 1),
                   max(0, c0 - 1):min(W, c0 + sw + 1)].any():
                continue
            if any(r0 + hr == 0 or c0 + hc == 0 or r0 + hr == H - 1 or c0 + hc == W - 1
                   for hpts in holes for hr, hc in hpts):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(grid, r0, c0, mask, color, fill, holes):
    """Render a sprite onto the grid: border cells in color, hole cells in fill (if given)."""
    grid[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _gen_attempt():
    rng = random.Random()
    bg = 0

    # Grid size and palette side
    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(['top', 'bottom', 'left', 'right'])
    if H <= _SMALL_H_THRESHOLD and side in ('top', 'bottom') and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(['left', 'right'])

    max_N = H if side in ('left', 'right') else W
    N = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_N))

    # Sprite dimension bounds
    max_sh = min(_SPRITE_H_MAX, H)
    max_sw = min(_SPRITE_W_MAX, W)
    if H <= _TINY_H_THRESHOLD:
        max_sh = 3

    # Sprite counts
    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, N))
    total_sprites = num_filled + 1

    # Generate sprite shapes
    shapes = []
    for _ in range(total_sprites):
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
        shapes.append(mask)

    # Colors and palette
    border_keys, fill_values, distractor_color, mapping, filled_keys = \
        _choose_colors(rng, N, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    # Assign colors to sprites: filled sprites use palette keys, last one is distractor
    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            c, f = filled_keys[i], mapping[filled_keys[i]]
        else:
            c, f = distractor_color, None
        infos.append((mask, c, f))

    # Place sprites
    placed = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(H, W, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill))
        sh, sw = mask.shape
        occ[max(0, r0 - 1):min(H, r0 + sh + 1),
            max(0, c0 - 1):min(W, c0 + sw + 1)] = True

    # Derive input from output by clearing filled holes
    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill in placed:
        if fill is not None:
            for hpts in _get_enclosed_holes(mask):
                for hr, hc in hpts:
                    inp[r0 + hr, c0 + hc] = bg

    # Validate bg is dominant on border
    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    # Oracle check
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input():
    for _ in range(_MAX_RETRIES):
        r = _gen_attempt()
        if r is not None:
            return r
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
