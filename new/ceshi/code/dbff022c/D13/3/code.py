# Variant of dbff022c
# Dimension: D13 — 填充颜色的确定来源 (transform)
# Change: 填充颜色完全由调色板图例显式给出... → 调色板和空间规则共同决定——位于调色板同侧的精灵使用正向映射，对侧使用反向映射

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis
# - Concrete behavior:
#   - The edge palette still occupies a 2xN or Nx2 block.
#   - The palette's outer line is the forward-key line and the inner line is the
#     forward-value line.
#   - A hollow sprite whose enclosed hole lies on the palette's side of the grid
#     uses the forward map outer->inner.
#   - A hollow sprite whose hole lies on the opposite half uses the reverse map
#     inner->outer.
#   - Example with a top palette [2 4] / [6 7]:
#       same-side sprite border 2 fills with 6
#       opposite-side sprite border 6 fills with 2
#     The difference from the original task is that the far sprite no longer uses
#     the palette's outer line as its key set.
# - Complexity calibration:
#   - Reverse lookup only makes sense if the two palette lines are both unique and
#     disjoint enough for humans to read them as two role-bearing sets.
#   - The generator therefore uses disjoint outer/inner colors and always creates
#     at least one forward-filled sprite and one reverse-filled sprite.
#   - A paired forward/reverse palette entry is reused on both sides so the new
#     rule is visible within a single sample, not only across samples.
# - Original-code assumptions that break under D13:
#   - `_palette_mapping()` previously built only one border->fill dictionary.
#     That silently assumes every fillable sprite border color comes from the
#     outer palette line. The variant invalidates that assumption.
#   - `_fill_enclosed_bg_regions()` previously chose a fill solely from the
#     adjacent border color. It now must also classify the hole by side.
#   - `_best_palette()` previously only required outer-line uniqueness. Reverse
#     lookup additionally requires inner-line uniqueness, so the detector now
#     checks both lines.
# - Placement legality:
#   - The generator keeps the palette isolated from sprites as before.
#   - Filled sprites are placed so that each enclosed hole is clearly in either
#     the palette half or the opposite half under the same midpoint test used by
#     the solver; ties are avoided by construction.
#   - Distractor sprite colors are outside both palette key sets so they stay
#     unchanged regardless of side.
# - Visual observability and cleanliness:
#   - The rule is visible from position alone: a sprite near the palette reads the
#     palette top/outer-to-inner, while one far from it reads inner-to-outer.
#   - No ordering uses color ids. Only edge side, visible halves, and palette row
#     membership matter.
#   - The grid still contains only separated hollow sprites and one compact palette
#     block, so objects remain easy to parse by eye.


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
                fill_keys = [int(x) for x in seg_b.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(fill_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
                    continue
                if len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(set(fill_keys)), len(d), -(c1 - c0), nonbg)
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
                fill_keys = [int(x) for x in seg_b.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(fill_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
                    continue
                if len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(set(fill_keys)), len(d), -(c1 - c0), nonbg)
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
                fill_keys = [int(x) for x in seg_b.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(fill_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
                    continue
                if len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(set(fill_keys)), len(d), -(r1 - r0), nonbg)
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
                fill_keys = [int(x) for x in seg_b.tolist() if int(x) != bg]
                if len(border_keys) == 0 or len(fill_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
                    continue
                if len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(set(fill_keys)), len(d), -(r1 - r0), nonbg)
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


def _palette_mappings(g: np.ndarray, bg: int) -> tuple[str, dict[int, int], dict[int, int], np.ndarray]:
    """
    Returns (side, forward_map outer->inner, reverse_map inner->outer, palette_mask).
    """
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}

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
    else:  # right
        pal_mask[start:end, w - 2 : w] = True
        outer_line = g[start:end, w - 1]
        inner_line = g[start:end, w - 2]

    for outer, inner in zip(outer_line.tolist(), inner_line.tolist(), strict=False):
        outer = int(outer)
        inner = int(inner)
        if outer == bg or inner == bg:
            continue
        forward[outer] = inner
        reverse[inner] = outer

    return side, forward, reverse, pal_mask


def _component_uses_forward(pts: np.ndarray, side: str, shape: tuple[int, int]) -> bool:
    """Classify a hole by whether it lies on the palette half of the grid."""
    h, w = shape
    center_r = float(np.mean(pts[:, 0]))
    center_c = float(np.mean(pts[:, 1]))
    if side == "top":
        return center_r <= (h - 1) / 2.0
    if side == "bottom":
        return center_r >= (h - 1) / 2.0
    if side == "left":
        return center_c <= (w - 1) / 2.0
    return center_c >= (w - 1) / 2.0


def _fill_enclosed_bg_regions(
    out: np.ndarray,
    bg: int,
    side: str,
    forward: dict[int, int],
    reverse: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """
    Find enclosed background regions outside the palette.

    Holes on the palette half use the outer->inner mapping.
    Holes on the opposite half use the inner->outer mapping.
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

        mapping = forward if _component_uses_forward(pts, side, out.shape) else reverse
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
    Fill hollow sprites using an edge legend whose direction depends on the sprite's side.

    Input structure:
    - Background is the dominant border color.
    - One grid edge contains a compact 2xN or Nx2 palette with an outer line and an inner line.
    - Hollow sprites may use colors from either palette line; the decisive cue is whether each hole sits on the same half as the palette edge or on the opposite half.

    Rule:
    - Detect the palette block and read both mappings: outer->inner and inner->outer.
    - Find enclosed background regions that do not touch the grid boundary.
    - If a hole lies on the palette side of the grid, fill it by matching its border color through the outer->inner map; if it lies on the far side, use the inner->outer map instead.

    Key insight:
    - The palette is a two-way legend, and spatial position chooses which line acts as the key set for a given sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    side, forward, reverse, pal_mask = _palette_mappings(g, bg)
    assert forward and reverse, "main: palette mappings must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, side, forward, reverse, pal_mask)

    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_TOP_BOTTOM_MIN_H = 10

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 400
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

    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(rng: random.Random, n: int) -> tuple[list[int], list[int], int, dict[int, int], dict[int, int]]:
    """Pick disjoint palette colors so forward and reverse lookup are both unambiguous."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    need = 2 * n + 1
    chosen = all_c[:need]
    border_keys = chosen[:n]
    fill_values = chosen[n : 2 * n]
    distractor_color = chosen[-1]
    forward = dict(zip(border_keys, fill_values, strict=False))
    reverse = dict(zip(fill_values, border_keys, strict=False))
    return border_keys, fill_values, distractor_color, forward, reverse


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the given edge side."""
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


def _holes_match_side(
    holes: list[np.ndarray],
    r0: int,
    c0: int,
    side: str,
    shape: tuple[int, int],
    require_forward: bool,
) -> bool:
    """Check that every hole in the placed sprite lands on the intended side."""
    for hpts in holes:
        placed = hpts + np.array([r0, c0], dtype=int)
        if _component_uses_forward(placed, side, shape) != require_forward:
            return False
    return True


def _find_valid_positions(
    h: int,
    w: int,
    occ: np.ndarray,
    mask: np.ndarray,
    holes: list[np.ndarray],
    side: str | None = None,
    require_forward: bool | None = None,
) -> list[tuple[int, int]]:
    """Find all positions where mask fits without overlap and the holes satisfy side constraints."""
    sh, sw = mask.shape
    positions = []
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
            if side is not None and require_forward is not None:
                if not _holes_match_side(holes, r0, c0, side, (h, w), require_forward):
                    continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite onto the grid: border cells in color, hole cells in fill (if given)."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _sample_sprite(rng: random.Random, max_sh: int, max_sw: int) -> np.ndarray | None:
    """Sample one hollow sprite mask."""
    sh = rng.randint(3, max_sh)
    sw = rng.randint(3, max_sw)
    if sh + sw > _SPRITE_DIM_SUM_CAP:
        sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
    if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
        return _make_diamond()
    return _make_hollow_sprite(sh, sw, rng)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    side = rng.choice(["top", "bottom", "left", "right"])
    h_min = _TOP_BOTTOM_MIN_H if side in ("top", "bottom") else _GRID_H_MIN
    h = rng.randint(h_min, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, h - 2)
    max_sw = min(_SPRITE_W_MAX, w - 2)

    border_keys, fill_values, distractor_color, forward, reverse = _choose_colors(rng, n)
    pair_idx = rng.randrange(n)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    specs = [
        ("same", border_keys[pair_idx], forward[border_keys[pair_idx]], True),
        ("opposite", fill_values[pair_idx], reverse[fill_values[pair_idx]], True),
        ("distractor", distractor_color, None, False),
    ]

    placed = []
    for role, color, fill, side_sensitive in specs:
        mask = None
        for _ in range(40):
            cand = _sample_sprite(rng, max_sh, max_sw)
            if cand is not None:
                mask = cand
                break
        if mask is None:
            return None

        holes = _get_enclosed_holes(mask)
        if role == "same":
            positions = _find_valid_positions(h, w, occ, mask, holes, side=side, require_forward=True)
        elif role == "opposite":
            positions = _find_valid_positions(h, w, occ, mask, holes, side=side, require_forward=False)
        else:
            positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None

        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill, side_sensitive))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill, side_sensitive in placed:
        if fill is not None and side_sensitive:
            for hpts in _get_enclosed_holes(mask):
                for hr, hc in hpts:
                    inp[r0 + hr, c0 + hc] = bg

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
