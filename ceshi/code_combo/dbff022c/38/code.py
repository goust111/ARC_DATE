# Combo variant of dbff022c (index 38)
# combo_spec: D2/3+D3/2+D7/2
# Name: 右边缘调色板 + 反直觉方向 + 自填充

import random

import numpy as np

try:
    from solvers.arc_utils import components_mask, neighbors4
except ImportError:
    from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace:
#    Original dbff022c fills each enclosed background hole only when its surrounding border
#    color appears in an edge palette, and the palette is read with the usual edge-based
#    direction rule. In this combo, the palette is always an N x 2 strip on the RIGHT edge,
#    and it must be read from RIGHT to LEFT: the outermost right column gives border keys,
#    the inner column gives fill colors. Example: if the right edge contains row pairs
#    [(6,3), (8,4), (2,5)] across columns [w-2, w-1], the true mapping is 3->6, 4->8, 5->2.
#    A border-3 hollow sprite fills with 6. A border-7 hollow sprite is not in the palette,
#    so the original task would leave its hole as 0, but this combo self-fills it with 7.
#    A left-to-right reader would wrongly infer 6->3, 8->4, 2->5, so the right-edge placement
#    and reversed reading direction must work together.
# b) Complexity calibration:
#    To keep the inversion visible, palette keys, palette fill colors, and the unmatched
#    sprite color are pairwise disjoint. This prevents accidental agreement between the true
#    mapping and the intuitive-but-wrong left-to-right parse. Each sample contains at least
#    one mapped sprite and exactly one unmatched sprite, so both "lookup fill" and
#    "self-fill fallback" are observable in the same grid without adding extra rule branches.
# c) Hidden assumptions review:
#    - Palette detection can no longer scan all four edges equally; the combo assumes the
#      only valid legend is a right-edge N x 2 block, so the solver now searches only there.
#    - The palette parser must treat column w-1 as the border-key line and column w-2 as the
#      fill line. Using the old left-to-right intuition would invert the mapping.
#    - Hole filling may not filter adjacent colors through `mapping` before choosing the
#      enclosing border, because unmatched colors must still be selected and then self-mapped
#      via `mapping.get(border, border)`.
#    - In-place hole filling remains safe because internal background components are separated
#      from one another by sprite borders, and the generator preserves at least one-cell gaps.
# d) Placement legality:
#    The generator fixes the palette at the right edge, marks a one-cell padded occupancy
#    band around it, and places hollow sprites only in non-overlapping padded positions.
#    Holes are checked not to touch the outer grid boundary, matching the solver's notion of
#    "enclosed background region".
# e) Visual observability:
#    The right-edge strip is visually distinct from the sprites, and disjoint color sets make
#    it easy to see which palette column matches border colors seen in the scene. Because one
#    unmatched sprite always becomes solid in the output, the D7/2 behavior is guaranteed to
#    differ from the original task on every generated sample.
# f) Cleanliness:
#    The grid still contains only three element types: background, one right-edge palette,
#    and separated hollow sprites. No extra markers or overlays are introduced.
# g) Inferability of change:
#    Color vocabulary is kept small: 2-4 palette pairs, 1-2 mapped sprites, and 1 unmatched
#    sprite. The structure, not raw color ids, signals the rule: sprite border colors reappear
#    only in the palette's OUTER right column, while output fills come from the inner column
#    or from self-filling when no palette row matches.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_right_palette(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Find the right-edge N x 2 palette block.

    Returns (start, end) as row indices [start, end).
    """
    h, w = g.shape
    assert w >= 2, "_best_right_palette requires at least 2 columns"
    best = None  # (score_tuple, start, end)

    def isolated(start: int, end: int) -> bool:
        pal = np.zeros((h, w), dtype=bool)
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
            merged = outer_colors | inner_colors
            if len(merged) < 3:
                continue
            border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
            if not border_keys or len(border_keys) != len(set(border_keys)):
                continue
            if not isolated(r0, r1):
                continue
            nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
            score = (len(border_keys), len(merged), nonbg, r1 - r0)
            cand = (score, r0, r1)
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None, "_best_right_palette must find a valid right-edge palette"
    _, start, end = best
    return int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Return (mapping border->fill, palette_mask) for the right-edge reversed palette.
    """
    h, w = g.shape
    start, end = _best_right_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    pal_mask[start:end, w - 2 : w] = True

    border_line = g[start:end, w - 1]
    fill_line = g[start:end, w - 2]
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
    """
    Fill each internal background region using its dominant adjacent border color.

    Mapped borders use the palette's fill color; unmatched borders self-fill.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    comps = components_mask(bg_mask, connectivity=4)
    for pts in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
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

        border = max(adj_counts.items(), key=lambda item: item[1])[0]
        fill = int(mapping.get(border, border))
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by reading a reversed legend that sits on the grid's right edge.

    Input structure:
    - Background is the most common border color, and several hollow single-color sprites are
      separated from one another and from the palette by at least one background cell.
    - The palette is an N x 2 strip on the right edge. Because this combo uses right-to-left
      reading, the OUTER right column lists border colors and the inner column lists fill colors.

    Rule:
    - Detect the right-edge palette and decode it as a border-color to fill-color lookup.
    - Find each enclosed background hole outside the palette area and identify the dominant
      border color surrounding that hole.
    - If that border color appears in the palette, fill with the paired inner-column color;
      otherwise fill with the border color itself.

    Key insight:
    - The right-edge legend is intentionally counterintuitive: sprite border colors match the
      outermost column, not the inner one. Once the border color around a hole is known, the
      transformation is either a palette lookup or a self-mapping fallback.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16

# Sprite mutation probabilities
_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

# Generation
_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point arrays for background components that stay inside the sprite mask."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped hollow mask."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = True
    s[1, 0] = True
    s[1, 2] = True
    s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a hollow sprite with at least one enclosed hole and bounded area."""
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        start = rng.randint(1, w - n - 1)
        sprite[br, start : start + n] = True
    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        start = rng.randint(1, h - n - 1)
        sprite[start : start + n, bc] = True
    if rng.random() < _P_BORDER_REMOVAL:
        border_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(border_cells)
        for cr, cc in border_cells[: rng.randint(1, min(4, len(border_cells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(
    rng: random.Random, n_pairs: int, num_mapped: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick disjoint palette keys, palette fills, and one unmatched sprite color."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n_pairs]
    fill_values = colors[n_pairs : 2 * n_pairs]
    distractor_color = colors[2 * n_pairs]
    mapping = dict(zip(border_keys, fill_values))
    mapped_keys = rng.sample(border_keys, num_mapped)
    return border_keys, fill_values, distractor_color, mapping, mapped_keys


def _build_right_palette(border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the right-edge N x 2 palette so it reads right-to-left."""
    pal = np.zeros((len(border_keys), 2), dtype=int)
    pal[:, 0] = fill_values
    pal[:, 1] = border_keys
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, rng: random.Random) -> None:
    """Place the palette on the right edge and mark padded occupancy."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    pr0 = rng.randint(0, h - pal_h)
    pc0 = w - pal_w
    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[max(0, pr0 - 1) : min(h, pr0 + pal_h + 1), max(0, pc0 - 1) : min(w, pc0 + pal_w + 1)] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements where the sprite fits, stays separated, and keeps holes internal."""
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
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite border plus its output-state fill."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + hr, c0 + hc] = fill


def _sample_shape(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    """Sample one hollow sprite shape within the current grid budget."""
    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
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

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, h))

    num_mapped = 1
    if n_pairs >= 2 and h >= 6 and w >= 12 and rng.random() < 0.65:
        num_mapped = 2
    total_sprites = num_mapped + 1  # exactly one unmatched self-fill sprite

    shapes = []
    for _ in range(total_sprites):
        mask = _sample_shape(rng, h, w)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_color, mapping, mapped_keys = _choose_colors(
        rng, n_pairs, num_mapped
    )
    pal = _build_right_palette(border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, rng)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_mapped:
            color = mapped_keys[i]
            fill = mapping[color]
        else:
            color = distractor_color
            fill = distractor_color
        infos.append((mask, color, fill))

    placed = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask in placed:
        for hpts in _get_enclosed_holes(mask):
            for hr, hc in hpts:
                inp[r0 + hr, c0 + hc] = bg

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
