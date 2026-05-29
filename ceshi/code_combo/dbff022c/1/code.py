# Combo variant of dbff022c (index 1)
# combo_spec: D1/0+D4/0
# Name: 反向映射 + 实心精灵替换

"""
Design analysis for the fused D1/0 + D4/0 variant.

a) Concrete behavior and trace
   - Original task:
     an irregular one-color border encloses background 0, and the edge palette is read as
     border -> fill. The solver finds each enclosed bg hole and fills it with the mapped color.
   - Fused task:
     the border geometry stays irregular and enclosed, but the interior is already a solid
     non-bg color. The palette direction is reversed: outer line = interior-key color,
     inner line = border color. The solver uses the existing interior color as the lookup key,
     checks which border color it maps to, and recolors the entire enclosed interior region to
     that border color.
   - Small trace:
       original input: border 4 around 0-cells, palette says 4 -> 6, output turns hole to 6.
       fused input: border 4 around solid interior 6, palette says 6 -> 4, output turns that
       interior region to 4.
     The changed cue is therefore twofold: the palette is read in the opposite direction, and
     the operation is replacement of an existing solid region rather than filling background.

b) Complexity calibration
   - Keep the original-style irregular border masks and 2-4 total sprites so the task still
     looks like the dbff022c family instead of collapsing into trivial rectangles.
   - Limit the palette to 2-4 pairs so fill keys and border values can stay disjoint, making
     the inversion visually legible.
   - Require 1-3 active sprites plus exactly 1 distractor. This gives repeated evidence that
     "interior key on the palette outer line controls the rewrite" while still leaving unused
     palette entries and a negative example.

c) Hidden assumption audit against the original main()
   - Original `_palette_mapping` assumed outer line = border keys. That is false now; outer
     line must be treated as interior colors and inner line as border colors.
   - Original `_fill_enclosed_bg_regions` searched background components. That fails because
     the actionable regions are solid non-bg interiors.
   - Original adjacency logic inferred the target color from the surrounding border only. In
     the fused task, the surrounding border is structural evidence, but the lookup key is the
     interior color itself.
   - New solver assumption:
     a recolorable interior is a monochrome non-bg component that stays off the canvas edge,
     does not touch background directly, and is adjacent to exactly one surrounding border
     color. The generator must guarantee that.

d) Placement legality
   - Sprites are built from genuine hollow masks first, so every painted interior component is
     truly enclosed by the border and cannot leak to the canvas boundary.
   - Palette and sprites use the same occupancy mask with one-cell padding, preventing overlap
     and preserving palette isolation for `_best_palette`.
   - Interior repainting is generated output-first: output stores the post-replacement color,
     then input rewrites only active interior cells back to their palette-key colors.

e) Visual observability
   - The palette line touching the edge matches input interior colors, not the final border
     colors. That makes the reversed direction readable from examples.
   - Output visibly solidifies active sprites into a single border-colored blob, while the
     distractor keeps its contrasting interior. This exposes the replacement semantics.
   - Several active sprites can share the same rule template within one grid, so the solver can
     infer the pattern from only a few examples.

f) Visual cleanliness
   - Borders, interiors, and the edge palette remain spatially separated by background.
   - Interior-key colors are disjoint from border colors, so the "before" and "after" states are
     easy to distinguish without relying on numeric color identities.
   - The distractor interior color is kept out of the palette-key set to avoid noisy exceptions.

g) Color-change inferability
   - Every changed color comes with a structural cue: it is the only solid region fully enclosed
     by one border color.
   - The replacement target is not arbitrary; it is the border color paired with that interior
     color on the palette. Because the output interior becomes identical to the border, the
     mapping can be inferred from local before/after comparison plus the edge legend.

Post-change main() assumptions guaranteed by the generator
   - Each recolorable interior component is monochrome and adjacent to exactly one border color.
   - For active sprites, that border color equals the palette value mapped from the interior key.
   - Active interiors never touch background or the grid boundary, and the distractor interior
     color is absent from the reversed palette mapping.
"""

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import components_mask, neighbors4


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
        Palette cells must stay isolated from same-colored external cells so the detector
        does not absorb parts of sprites that happen to live on the boundary.
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
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                all_cols = d_outer | d_inner
                if len(all_cols) < 3:
                    continue
                fill_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not fill_keys or len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(fill_keys)), len(all_cols), -(c1 - c0), nonbg)
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
                all_cols = d_outer | d_inner
                if len(all_cols) < 3:
                    continue
                fill_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not fill_keys or len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(fill_keys)), len(all_cols), -(c1 - c0), nonbg)
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
                all_cols = d_outer | d_inner
                if len(all_cols) < 3:
                    continue
                fill_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not fill_keys or len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(fill_keys)), len(all_cols), -(r1 - r0), nonbg)
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
                all_cols = d_outer | d_inner
                if len(all_cols) < 3:
                    continue
                fill_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not fill_keys or len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(fill_keys)), len(all_cols), -(r1 - r0), nonbg)
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
    Returns (mapping interior_key->border_color, palette_mask).
    """
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)

    if side == "top":
        pal_mask[0:2, start:end] = True
        fill_line = g[0, start:end]
        border_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        fill_line = g[h - 1, start:end]
        border_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        fill_line = g[start:end, 0]
        border_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        fill_line = g[start:end, w - 1]
        border_line = g[start:end, w - 2]

    mapping: dict[int, int] = {}
    for fill_key, border in zip(fill_line.tolist(), border_line.tolist(), strict=False):
        fill_key = int(fill_key)
        border = int(border)
        if fill_key == bg:
            continue
        mapping[fill_key] = border
    return mapping, pal_mask


def _recolor_reversed_interior_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Recolor enclosed non-bg regions using their own color as the palette key.

    A region is recolorable when it:
      - lies outside the palette,
      - does not touch the grid boundary,
      - does not touch background directly,
      - is monochrome by construction,
      - is adjacent to exactly one non-bg color,
      - and that enclosing color matches mapping[region_color].
    """
    h, w = out.shape
    candidate_colors = [int(c) for c in np.unique(out[~pal_mask]) if int(c) != bg]

    for color in candidate_colors:
        comps = components_mask((out == color) & (~pal_mask), connectivity=4)
        for pts in comps:
            if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
                continue

            adj_counts: dict[int, int] = {}
            touches_bg = False
            for r, c in pts:
                for rr, cc in neighbors4(int(r), int(c)):
                    if not (0 <= rr < h and 0 <= cc < w):
                        touches_bg = True
                        break
                    if pal_mask[rr, cc]:
                        continue
                    v = int(out[rr, cc])
                    if v == color:
                        continue
                    if v == bg:
                        touches_bg = True
                        break
                    adj_counts[v] = adj_counts.get(v, 0) + 1
                if touches_bg:
                    break
            if touches_bg or len(adj_counts) != 1:
                continue
            if color not in mapping:
                continue

            border = next(iter(adj_counts))
            target = int(mapping[color])
            if border != target:
                continue

            for r, c in pts:
                out[int(r), int(c)] = target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor solid sprite interiors using a reversed color legend on the grid edge.

    Input structure:
    - The dominant edge color is the background, and a compact 2xN or Nx2 palette block sits on one grid edge.
    - The palette is reversed relative to the original task family: its outer row/column lists interior-key colors, and its inner row/column lists the paired border colors.
    - Away from the palette are irregular single-color border sprites whose enclosed interiors are already solid non-bg regions.

    Rule:
    - Detect the edge palette and read it as interior-color -> border-color.
    - Find enclosed non-bg interior components outside the palette that stay off the outer boundary and are surrounded by exactly one border color.
    - If an interior component's own color appears in the reversed palette and the mapped border color matches its enclosing border, replace the whole interior with that border color. Components whose interior color is absent from the palette remain unchanged.

    Key insight:
    - The useful lookup cue is no longer the border color and no longer an empty hole. The solver must use the existing solid interior color as the palette key, reverse the legend direction, and then rewrite that interior so it merges into the surrounding border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _recolor_reversed_interior_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
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
_ACTIVE_SPRITE_MAX = 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 500
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components inside a border mask that do not touch the mask bbox boundary."""
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
    """Create an irregular border mask with at least one enclosed hole."""
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
    rng: random.Random, n: int, num_active: int
) -> tuple[list[int], list[int], dict[int, int], list[int], int, int]:
    """
    Choose reversed-palette colors.

    fill_keys live on the palette outer line and appear inside active sprites in the input.
    border_values live on the palette inner line and become the surrounding / replacement color.
    """
    colors = list(range(1, 10))
    rng.shuffle(colors)
    fill_keys = colors[:n]
    border_values = colors[n : 2 * n]
    mapping = dict(zip(fill_keys, border_values))
    active_fill_keys = rng.sample(fill_keys, num_active)

    remaining = colors[2 * n :]
    distractor_interior = remaining[0]
    distractor_border = remaining[1] if len(remaining) > 1 else rng.choice(border_values)
    return fill_keys, border_values, mapping, active_fill_keys, distractor_interior, distractor_border


def _build_palette(side: str, fill_keys: list[int], border_values: list[int]) -> np.ndarray:
    """Construct the reversed 2xN or Nx2 palette for the given edge side."""
    n = len(fill_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = fill_keys
        pal[inner] = border_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = fill_keys
        pal[:, inner] = border_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place palette on the chosen edge and mark its occupancy with one-cell padding."""
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
    """Find placements where the border mask fits and all interior cells stay off the outer border."""
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
    border_color: int,
    interior_color: int,
    holes: list[np.ndarray],
) -> None:
    """Render a border mask plus solid enclosed interior cells."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = border_color
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + int(hr), c0 + int(hc)] = interior_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    num_active = rng.randint(1, min(_ACTIVE_SPRITE_MAX, n))
    total_sprites = num_active + 1

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

    fill_keys, border_values, mapping, active_fill_keys, distractor_interior, distractor_border = _choose_colors(
        rng, n, num_active
    )
    pal = _build_palette(side, fill_keys, border_values)

    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        holes = _get_enclosed_holes(mask)
        if i < num_active:
            interior_key = active_fill_keys[i]
            border_color = mapping[interior_key]
            output_interior = border_color
            input_interior = interior_key
            active = True
        else:
            border_color = distractor_border
            output_interior = distractor_interior
            input_interior = distractor_interior
            active = False
        infos.append((mask, holes, border_color, output_interior, input_interior, active))

    placed = []
    for mask, holes, border_color, output_interior, input_interior, active in infos:
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(output, r0, c0, mask, border_color, output_interior, holes)
        placed.append((r0, c0, holes, input_interior, active))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    inp = output.copy()
    changed_cells = 0
    for r0, c0, holes, input_interior, active in placed:
        if not active:
            continue
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = input_interior
                changed_cells += 1
    if changed_cells == 0:
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
    rng = random.Random()
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
