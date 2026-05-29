# Combo variant of dbff022c (index 33)
# combo_spec: D2/0+D3/0
# Name: 角落调色板 + 固定阅读方向

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant
# 1. Concrete behavior trace:
#    A bottom-right horizontal palette now looks like:
#      [2 3 4]
#      [7 8 9]
#    placed in the corner at rows H-2:H, cols W-3:W. The original task would
#    read the outer bottom row as the border-key line, giving 7->2, 8->3, 9->4.
#    This combo instead always reads a horizontal palette by local coordinates:
#    top row = border keys, bottom row = fill values, left-to-right. So the same
#    corner palette means 2->7, 3->8, 4->9. A sprite outlined with color 3 must
#    therefore fill with 8, not remain empty and not map in reverse.
# 2. Complexity calibration:
#    Corner placement already increases search difficulty, so direction logic is
#    simplified to a single local rule. To avoid ambiguous 2x2 corner blocks that
#    could be parsed both horizontally and vertically, palette length is restricted
#    to 3-4 entries. Border colors and fill colors are generated from disjoint sets,
#    which makes the mapping legible and ensures fixed-direction cases differ from
#    the original edge-dependent reading.
# 3. Broken implicit assumptions from the seed solver:
#    The seed code assumes the palette sits somewhere along one edge and that bottom
#    and right palettes must be read "from the outside inward". That assumption is
#    no longer valid. The new detector only inspects corner-anchored 2xN/Nx2 blocks
#    and always interprets the first local row/column as border colors.
# 4. Placement legality:
#    The generator anchors the palette flush to one of the four corners, reserves a
#    one-cell padding halo, and only places sprites in non-overlapping positions whose
#    holes stay away from the global boundary. This keeps the palette and sprites
#    visually separate and preserves the "enclosed hole" notion needed by the solver.
# 5. Visual observability:
#    The combination is human-visible because the palette is clearly in a corner and
#    fixed-direction cases differ most strongly for bottom and right corners. Since
#    border and fill colors are disjoint, reversing the readout would obviously fail
#    to explain the filled interiors in those examples.
# 6. Visual cleanliness:
#    Only one palette exists, it has a bg moat around it, sprites are separated by at
#    least one bg cell, and there is exactly one distractor sprite whose hole stays
#    empty. This avoids layered clutter and keeps the mapping cue easy to spot.
# 7. New solver assumptions audited against the generator:
#    - `_best_palette` assumes the palette is corner-anchored, rectangular, all non-bg,
#      and isolated from same-colored neighbors outside the palette. The generator
#      guarantees each of these conditions.
#    - `_fill_enclosed_bg_regions` assumes holes do not touch the outer boundary and
#      their dominant adjacent color is the sprite border color. The sprite placement
#      and output-first construction preserve that property.


_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
_ORIENTATIONS = ("horizontal", "vertical")


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_bounds(
    shape: tuple[int, int], corner: str, orientation: str, n: int
) -> tuple[int, int, int, int]:
    h, w = shape
    if orientation == "horizontal":
        ph, pw = 2, n
        r0 = 0 if corner.startswith("top") else h - ph
        c0 = 0 if corner.endswith("left") else w - pw
    else:
        ph, pw = n, 2
        r0 = 0 if corner.startswith("top") else h - ph
        c0 = 0 if corner.endswith("left") else w - pw
    return int(r0), int(c0), int(ph), int(pw)


def _isolated_palette(g: np.ndarray, bg: int, pal_mask: np.ndarray) -> bool:
    """Palette colors must not 4-touch identical colors outside the palette."""
    h, w = g.shape
    for r, c in np.argwhere(pal_mask):
        r = int(r)
        c = int(c)
        v = int(g[r, c])
        if v == bg:
            return False
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                if int(g[rr, cc]) == v:
                    return False
    return True


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, str, int, int, int, np.ndarray]:
    """
    Find the corner-anchored 2xN or Nx2 palette block.

    Returns (corner, orientation, n, r0, c0, pal_mask).
    """
    h, w = g.shape
    best = None  # (score, corner, orientation, n, r0, c0, pal_mask)

    for corner in _CORNERS:
        for orientation in _ORIENTATIONS:
            max_n = w if orientation == "horizontal" else h
            for n in range(3, max_n + 1):
                r0, c0, ph, pw = _palette_bounds((h, w), corner, orientation, n)
                block = g[r0 : r0 + ph, c0 : c0 + pw]
                if np.any(block == bg):
                    continue

                if orientation == "horizontal":
                    border_line = [int(x) for x in block[0, :].tolist()]
                    fill_line = [int(x) for x in block[1, :].tolist()]
                else:
                    border_line = [int(x) for x in block[:, 0].tolist()]
                    fill_line = [int(x) for x in block[:, 1].tolist()]

                if len(set(border_line)) != n or len(set(fill_line)) != n:
                    continue

                pal_mask = np.zeros((h, w), dtype=bool)
                pal_mask[r0 : r0 + ph, c0 : c0 + pw] = True
                if not _isolated_palette(g, bg, pal_mask):
                    continue

                score = (n, len(set(border_line) | set(fill_line)))
                cand = (score, corner, orientation, n, r0, c0, pal_mask)
                if best is None or cand[0] > best[0]:
                    best = cand

    assert best is not None, "_best_palette must find a valid corner palette"
    _, corner, orientation, n, r0, c0, pal_mask = best
    return corner, orientation, int(n), int(r0), int(c0), pal_mask


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return (mapping border->fill, palette_mask)."""
    _, orientation, n, r0, c0, pal_mask = _best_palette(g, bg)

    if orientation == "horizontal":
        border_line = g[r0, c0 : c0 + n]
        fill_line = g[r0 + 1, c0 : c0 + n]
    else:
        border_line = g[r0 : r0 + n, c0]
        fill_line = g[r0 : r0 + n, c0 + 1]

    mapping: dict[int, int] = {}
    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        mapping[int(b)] = int(f)
    return mapping, pal_mask


def _fill_enclosed_bg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """Fill non-boundary bg holes using the dominant adjacent mapped border color."""
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
                    if v != bg:
                        adj_counts[v] = adj_counts.get(v, 0) + 1

        border = None
        best = -1
        for color, cnt in adj_counts.items():
            if color in mapping and cnt > best:
                best = cnt
                border = color
        if border is None:
            continue

        fill = mapping[border]
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by reading a corner palette that uses one fixed local direction.

    Input structure:
    - The background dominates the outer border.
    - Several hollow single-color sprites sit away from the edges.
    - One palette block is anchored flush to a grid corner. It is either 2xN or Nx2,
      with no background cells inside it. Horizontal palettes always mean
      "top row = border colors, bottom row = fill colors"; vertical palettes always
      mean "left column = border colors, right column = fill colors".

    Rule:
    - Locate the palette among the four corners and decode the border->fill mapping
      using the fixed local reading convention, not the global edge side.
    - Find background regions that are enclosed away from the outer boundary.
    - For each such region, inspect the surrounding sprite color; if that color is a
      palette key, fill the region with the mapped palette value.

    Key insight:
    - The only positional challenge is identifying which corner contains the legend.
      Once found, the palette is read in the same local orientation everywhere, so the
      decisive step is matching each sprite border color to the palette key line.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (combo) ---------------

_GRID_H_MIN, _GRID_H_MAX = 5, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15

_PALETTE_N_MIN, _PALETTE_N_MAX = 3, 4

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_SHAPE_TRIES = 32
_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point-arrays for bg components not touching the sprite boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
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


def _sample_shape(max_sh: int, max_sw: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(_MAX_SHAPE_TRIES):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            return _make_diamond()
        mask = _make_hollow_sprite(sh, sw, rng)
        if mask is not None:
            return mask
    return None


def _choose_colors(
    rng: random.Random, n: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick disjoint border/fill colors plus one distractor color."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n]
    fill_values = colors[n : 2 * n]
    distractor_color = colors[2 * n]
    mapping = dict(zip(border_keys, fill_values, strict=False))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(orientation: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct a palette using the fixed local reading direction."""
    n = len(border_keys)
    if orientation == "horizontal":
        pal = np.zeros((2, n), dtype=int)
        pal[0, :] = border_keys
        pal[1, :] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        pal[:, 0] = border_keys
        pal[:, 1] = fill_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, corner: str) -> None:
    """Place palette flush to the selected corner and mark a 1-cell padding halo."""
    h, w = grid.shape
    ph, pw = pal.shape
    r0 = 0 if corner.startswith("top") else h - ph
    c0 = 0 if corner.endswith("left") else w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements with padding and with all hole cells away from the boundary."""
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
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite: border in `color`, optional hole fill in `fill`."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + int(hr), c0 + int(hc)] = fill


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    orientation = rng.choice(_ORIENTATIONS)
    corner = rng.choice(_CORNERS)
    n = rng.randint(_PALETTE_N_MIN, _PALETTE_N_MAX)

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n))
    total_sprites = num_filled + 1

    shapes = []
    for _ in range(total_sprites):
        mask = _sample_shape(max_sh, max_sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n, num_filled)
    pal = _build_palette(orientation, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, corner)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            color = filled_keys[i]
            fill = mapping[color]
        else:
            color = distractor_color
            fill = None
        infos.append((mask, color, fill))
    infos.sort(key=lambda item: int(item[0].shape[0] * item[0].shape[1]), reverse=True)

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

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill in placed:
        if fill is None:
            continue
        for hpts in _get_enclosed_holes(mask):
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = bg

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
