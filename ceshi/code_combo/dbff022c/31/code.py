# Combo variant of dbff022c (index 31)
# combo_spec: D4/0+D5/4+D7/2
# Name: 实心精灵替换 + 颜色互换 + 自填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis
# a) Concrete behavior and trace
# - The source task looks for enclosed background holes and fills them from an edge palette.
# - This combo changes the actionable object completely: each sprite interior is already a solid
#   non-background region in the input. Example trace:
#   palette says 4->6, and there is a sprite whose input border is 4 and whose input interior is a
#   placeholder color 5. Original dbff022c would do nothing here because there is no internal 0-hole.
#   In this combo, the sprite is still "matched" by border color 4, so the output recolors the
#   entire border to 6 and rewrites the interior to 4. A second sprite with border 7 and the same
#   placeholder interior 5 is not listed in the palette; its output keeps border 7 and changes the
#   interior to 7, producing a solid 7-block. The key difference from both the source task and the
#   pure D4 variant is that mapped sprites do a two-way swap, while unmapped sprites still replace
#   the pre-existing interior but default to self-fill.
#
# b) Complexity calibration
# - A single shared placeholder interior color is used across sprites so humans can see that the
#   solver is rewriting an already-filled region, not discovering zeros.
# - Every generated sample contains both cases: at least one palette-matched swap sprite and exactly
#   one unmapped self-fill sprite. That prevents degeneration into "always swap" or "always recolor".
# - Border colors are unique per sprite and sprites stay separated by a one-cell moat, so the border
#   color surrounding an interior component is visually and algorithmically unambiguous.
#
# c) Hidden assumptions audit
# - The original fill loop assumed the target region color was background; that assumption is false
#   here, so the solver must iterate over non-background components outside the palette.
# - D5-style swapping adds a second broken assumption: changing only the interior is insufficient.
#   Once a matched sprite is recognized, the entire enclosing border component must be recolored too.
# - The solver now assumes each actionable interior component is enclosed by exactly one non-bg color
#   and does not touch outside background. The generator guarantees this by building connected hollow
#   border masks first, then replacing their enclosed holes with solid interior cells.
# - The border-component flood assumes a sprite border is a single 4-connected monochrome component.
#   The generator validates that property after all shape mutations.
#
# d) Placement legality
# - Palette placement still reserves a padded occupancy zone on one outer edge.
# - Each sprite placement checks padded non-overlap and also checks that every enclosed interior cell
#   stays off the full-grid boundary. That keeps the solver's "internal region" test aligned with
#   the generator's construction.
# - Output is generated first, then input is derived by restoring each matched sprite's original
#   border color and overwriting every interior cell with the placeholder color.
#
# e) Visual observability
# - Across 2-4 examples, a human sees three repeated visual roles: the palette on the edge, solid
#   placeholder interiors in the input, and two output outcomes inside the sprite field:
#   matched sprites swap border/interior colors, unmatched sprites become border-colored solids.
# - No branch uses numeric ordering of colors. Matching is visible only through equality between a
#   sprite border color and a palette key.
#
# f) Visual cleanliness
# - The scene still contains only three element types: background, one edge palette, and separated
#   monochrome-bordered sprites with solid interiors. No extra markers are layered on top.
# - Using a shared placeholder color keeps inputs tidy instead of introducing many unrelated interior
#   colors, while the outputs remain easy to parse because each sprite changes in a uniform way.
#
# g) Change inferability
# - Color changes are supported by structure: the border color determines the transformation branch,
#   and the interior region is identified by enclosure rather than by its own hue.
# - The palette remains small, the number of active mapped sprites is at most three, and the single
#   unmapped sprite provides a consistent negative example for the default self-mapping rule.


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
            rr = int(r)
            cc = int(c)
            v = int(g[rr, cc])
            if v == bg:
                continue
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and not pal[nr, nc] and int(g[nr, nc]) == v:
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
                cols_outer = set(map(int, np.unique(seg_outer)))
                cols_inner = set(map(int, np.unique(seg_inner)))
                cols_outer.discard(bg)
                cols_inner.discard(bg)
                if len(cols_outer) < 2 or len(cols_inner) < 2:
                    continue
                all_cols = cols_outer | cols_inner
                if len(all_cols) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(all_cols), -(c1 - c0), nonbg), "top", c0, c1)
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
                cols_outer = set(map(int, np.unique(seg_outer)))
                cols_inner = set(map(int, np.unique(seg_inner)))
                cols_outer.discard(bg)
                cols_inner.discard(bg)
                if len(cols_outer) < 2 or len(cols_inner) < 2:
                    continue
                all_cols = cols_outer | cols_inner
                if len(all_cols) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(all_cols), -(c1 - c0), nonbg), "bottom", c0, c1)
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
                cols_outer = set(map(int, np.unique(seg_outer)))
                cols_inner = set(map(int, np.unique(seg_inner)))
                cols_outer.discard(bg)
                cols_inner.discard(bg)
                if len(cols_outer) < 2 or len(cols_inner) < 2:
                    continue
                all_cols = cols_outer | cols_inner
                if len(all_cols) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(all_cols), -(r1 - r0), nonbg), "left", r0, r1)
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
                cols_outer = set(map(int, np.unique(seg_outer)))
                cols_inner = set(map(int, np.unique(seg_inner)))
                cols_outer.discard(bg)
                cols_inner.discard(bg)
                if len(cols_outer) < 2 or len(cols_inner) < 2:
                    continue
                all_cols = cols_outer | cols_inner
                if len(all_cols) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(all_cols), -(r1 - r0), nonbg), "right", r0, r1)
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
    """Return (mapping original_border->replacement_border, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
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

    mapping: dict[int, int] = {}
    for border, repl in zip(outer_line.tolist(), inner_line.tolist(), strict=False):
        b = int(border)
        f = int(repl)
        if b == bg:
            continue
        mapping[b] = f

    return mapping, pal_mask


def _collect_component_from_seeds(
    g: np.ndarray,
    pal_mask: np.ndarray,
    color: int,
    seeds: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Collect the full 4-connected monochrome component reached from the seed set."""
    h, w = g.shape
    stack = [(int(r), int(c)) for r, c in seeds]
    seen: set[tuple[int, int]] = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        if not (0 <= r < h and 0 <= c < w):
            continue
        if pal_mask[r, c] or int(g[r, c]) != int(color):
            continue
        seen.add((r, c))
        for nr, nc in neighbors4(r, c):
            stack.append((int(nr), int(nc)))
    return seen


def _swap_or_self_fill_solid_interiors(
    inp: np.ndarray,
    out: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """
    Find enclosed solid interior components and transform them by their surrounding border color.

    Matched sprite:
      border b with palette entry b->f  => border becomes f, interior becomes b

    Unmatched sprite:
      border b absent from palette      => border stays b, interior becomes b
    """
    h, w = inp.shape
    colors = [int(c) for c in np.unique(inp[~pal_mask]) if int(c) != bg]

    for color in colors:
        comps = components_mask((inp == color) & (~pal_mask), connectivity=4)
        for pts in comps:
            if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
                continue

            touches_bg = False
            adj_counts: dict[int, int] = {}
            seeds_by_color: dict[int, set[tuple[int, int]]] = {}
            for r, c in pts:
                rr = int(r)
                cc = int(c)
                for nr, nc in neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        touches_bg = True
                        break
                    if pal_mask[nr, nc]:
                        continue
                    v = int(inp[nr, nc])
                    if v == color:
                        continue
                    if v == bg:
                        touches_bg = True
                        break
                    adj_counts[v] = adj_counts.get(v, 0) + 1
                    seeds_by_color.setdefault(v, set()).add((int(nr), int(nc)))
                if touches_bg:
                    break
            if touches_bg or len(adj_counts) != 1:
                continue

            border = next(iter(adj_counts))
            border_component = _collect_component_from_seeds(
                inp,
                pal_mask,
                border,
                sorted(seeds_by_color[border]),
            )
            if not border_component:
                continue

            new_border = int(mapping.get(border, border))
            for r, c in border_component:
                out[r, c] = new_border
            for r, c in pts:
                out[int(r), int(c)] = int(border)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap palette-matched solid sprites and self-fill unmatched ones.

    Input structure:
    - The dominant outer-border color is the background, and one edge holds a 2xN or Nx2 palette.
    - The palette's outer strip lists sprite border colors; its inner strip lists the border colors
      that matched sprites should change into.
    - Away from the palette sit irregular monochrome-bordered sprites whose enclosed interiors are
      already filled with a solid non-background placeholder color.

    Rule:
    - Read the edge palette as original-border -> replacement-border.
    - Find each enclosed solid interior region outside the palette by looking for a non-background
      component that does not touch the canvas edge, does not touch background, and is surrounded by
      exactly one non-background border color.
    - If that border color is in the palette, recolor the whole enclosing border component to the
      mapped color and recolor the interior to the sprite's original border color. If the border
      color is absent from the palette, leave the border as-is and recolor the interior to that same
      border color.

    Key insight:
    - The interior is no longer an empty hole. The solver must first recognize an already-colored
      enclosed region, then use the surrounding border color to choose between a true border/interior
      color swap and a default self-fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _swap_or_self_fill_solid_interiors(g, out, bg, mapping, pal_mask)
    return out


# --------------- generator ---------------

_GRID_H_MIN, _GRID_H_MAX = 4, 13
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
_P_BORDER_REMOVAL = 0.35
_P_DIAMOND = 0.2

_MAX_RETRIES = 500


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return all bg components inside the sprite bbox that do not touch the bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped border mask with a single enclosed center cell."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _is_valid_swap_sprite(sprite: np.ndarray) -> bool:
    """Connected monochrome border with at least one enclosed hole and bounded cell count."""
    if int(sprite.sum()) > _SPRITE_CELL_MAX:
        return False
    if len(components_mask(sprite, connectivity=4)) != 1:
        return False
    return len(_get_enclosed_holes(sprite)) >= 1


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create an irregular but connected hollow sprite suitable for border recoloring."""
    if h == 3 and w == 3 and rng.random() < _P_DIAMOND:
        return _make_diamond()

    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        span = rng.randint(1, w - 2)
        start = rng.randint(1, w - span - 1)
        sprite[br, start : start + span] = True
    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        span = rng.randint(1, h - 2)
        start = rng.randint(1, h - span - 1)
        sprite[start : start + span, bc] = True

    if rng.random() < _P_BORDER_REMOVAL:
        edge_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(edge_cells)
        for cr, cc in edge_cells[: rng.randint(1, min(4, len(edge_cells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _is_valid_swap_sprite(test):
                sprite = test

    if not _is_valid_swap_sprite(sprite):
        return None
    return sprite


def _choose_colors(
    rng: random.Random,
    n_pairs: int,
    num_active: int,
) -> tuple[list[int], list[int], int, int, dict[int, int], list[int]]:
    """
    Pick palette keys, palette replacement colors, one shared interior placeholder,
    one unmapped border color, and the active palette keys used by matched sprites.
    """
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n_pairs]
    remaining = [c for c in colors if c not in border_keys]

    placeholder = remaining[0]
    fill_pool = [c for c in remaining if c != placeholder]
    rng.shuffle(fill_pool)
    fill_values = fill_pool[:n_pairs]

    distractor_candidates = [c for c in remaining if c not in fill_values and c != placeholder]
    if distractor_candidates:
        distractor_border = rng.choice(distractor_candidates)
    else:
        distractor_border = rng.choice([c for c in remaining if c != placeholder])

    mapping = dict(zip(border_keys, fill_values))
    active_keys = rng.sample(border_keys, num_active)
    return border_keys, fill_values, placeholder, distractor_border, mapping, active_keys


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
    grid: np.ndarray,
    occ: np.ndarray,
    pal: np.ndarray,
    side: str,
    rng: random.Random,
) -> None:
    """Place palette on the edge and reserve a one-cell padded occupancy region."""
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
    h: int,
    w: int,
    occ: np.ndarray,
    mask: np.ndarray,
    holes: list[np.ndarray],
) -> list[tuple[int, int]]:
    """Find padded non-overlapping placements whose enclosed interior cells stay off the canvas edge."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + int(hr) == 0
                or c0 + int(hc) == 0
                or r0 + int(hr) == h - 1
                or c0 + int(hc) == w - 1
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
    border_color: int,
    interior_color: int,
    holes: list[np.ndarray],
) -> None:
    """Render a bordered sprite whose enclosed hole cells are filled with one solid color."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(border_color)
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + int(hr), c0 + int(hc)] = int(interior_color)


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

    num_active = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))
    total_sprites = num_active + 1

    shapes = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, placeholder, distractor_border, mapping, active_keys = _choose_colors(
        rng, n_pairs, num_active
    )
    pal = _build_palette(side, border_keys, fill_values)

    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        holes = _get_enclosed_holes(mask)
        if i < num_active:
            input_border = int(active_keys[i])
            output_border = int(mapping[input_border])
            output_interior = int(input_border)
            is_matched = True
        else:
            input_border = int(distractor_border)
            output_border = int(distractor_border)
            output_interior = int(distractor_border)
            is_matched = False
        infos.append((mask, holes, input_border, output_border, output_interior, is_matched))

    placed = []
    for mask, holes, input_border, output_border, output_interior, is_matched in infos:
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(output, r0, c0, mask, output_border, output_interior, holes)
        placed.append((r0, c0, mask, holes, input_border, is_matched))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    inp = output.copy()
    swap_border_cells = 0
    self_fill_cells = 0
    for r0, c0, mask, holes, input_border, is_matched in placed:
        inp[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(input_border)
        if is_matched:
            swap_border_cells += int(mask.sum())
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = int(placeholder)
                if not is_matched:
                    self_fill_cells += 1

    if swap_border_cells == 0 or self_fill_cells == 0:
        return None

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
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
