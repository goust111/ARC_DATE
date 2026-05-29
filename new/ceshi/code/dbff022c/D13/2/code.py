# Variant of dbff022c
# Dimension: D13 — 填充颜色的确定来源 (transform)
# Change: 填充颜色完全由调色板图例显式给出... → 填充颜色由最近邻精灵的边框颜色决定

from __future__ import annotations

import random
import sys

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import components_mask


# Design analysis for the D13 variant
# a) Concrete behavior and trace:
#    - Keep the original visual grammar: a black background with 2-4 separated hollow
#      single-color sprites. Input shows empty holes; output fills each hole.
#    - The changed part is only the fill-color source. Originally, a hole looked at its
#      surrounding border color, then consulted an explicit edge palette to get the fill.
#      Now there is no palette. A red hollow sprite near a blue sprite and far from a yellow
#      sprite fills with blue, because blue is the nearest other sprite border color.
#    - Example trace:
#      old: hole inside red sprite -> adjacent border is red -> palette says red->green
#      -> fill hole green.
#      new: hole inside red sprite -> find the nearest other sprite in the scene
#      (suppose blue is closer than yellow) -> fill hole blue.
#      The ownership of a hole stays local to its enclosing sprite, but the output color is
#      now decided globally by sprite-to-sprite distance.
# b) Complexity calibration:
#    - Using 2-4 sprites keeps enough spatial relations for the rule to be visible without
#      creating a combinatorial mess.
#    - Border colors are unique per sprite so the transferred color is visually attributable.
#    - Generator rejects nearest-distance ties, so a sample never depends on an invisible
#      tie-break rule.
# c) Original assumptions that no longer hold:
#    - `_best_palette` / `_palette_mapping` assumed a unique 2xN or Nx2 edge legend. This is
#      removed because D13 explicitly forbids the palette mechanism.
#    - Original filling logic only needed the dominant adjacent border color of each enclosed
#      region; that still identifies which sprite owns a hole, but it no longer determines the
#      fill color by itself.
#    - The old `pal_mask` exclusion existed only to avoid filling inside the legend area. With
#      no legend, all non-boundary-touching bg holes are candidate interiors.
# d) Placement legality:
#    - Sprites remain separated from one another by at least one bg cell using the original
#      occupancy-padding strategy, so objects never overlap and nearest distances are well-defined.
#    - Holes are generated strictly inside sprite bboxes, so clearing them for the input does not
#      merge sprites or create stray bg tunnels to the boundary.
# e) Visual observability:
#    - The output color always matches another visible sprite border, never a hidden numeric rule.
#    - With 3-4 sprites, different targets can inherit different neighbor colors, making the new
#      spatial dependency readable from examples.
# f) Visual cleanliness:
#    - No palette block remains on the edge, which reduces clutter and keeps the scene focused on
#      hollow sprites and their spacing.
#    - Unique sprite colors and one-cell spacing prevent ambiguous overlaps.
# g) Color-change inferability:
#    - At most four sprite colors appear, and every fill color is copied directly from another
#      sprite already present in the same grid. This provides a concrete visual source for each
#      change.
# h) New solver assumptions audited against the generator:
#    - Solver groups sprite cells with 8-connectivity so the optional 3x3 diamond shape counts as
#      one sprite; generator may emit that shape, so this assumption is necessary and satisfied.
#    - Solver assumes each sprite has a unique nearest neighbor under minimum Manhattan distance;
#      generator checks this and rejects tied layouts.
#    - Solver fills every enclosed hole of a sprite with one color, so generator keeps each sprite
#      monochromatic on its border and never mixes multiple source rules within one sprite.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return local bg components that stay strictly inside the sprite bbox."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _extract_sprites(g: np.ndarray, bg: int) -> list[dict]:
    """Extract monochromatic sprites, their local holes, and their global coordinates."""
    sprites: list[dict] = []
    for color in sorted(set(map(int, g.flatten())) - {bg}):
        for comp in components_mask(g == color, connectivity=8):
            coords = np.asarray(comp, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in coords:
                mask[int(r) - r0, int(c) - c0] = True
            holes = []
            for pts in _get_enclosed_holes(mask):
                holes.append(np.asarray([(r0 + int(r), c0 + int(c)) for r, c in pts], dtype=int))
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "holes": holes,
                }
            )
    return sprites


def _sprite_distance(a: dict, b: dict) -> int:
    """Minimum Manhattan distance between two sprites' colored cells."""
    ac = a["coords"]
    bc = b["coords"]
    dr = np.abs(ac[:, None, 0] - bc[None, :, 0])
    dc = np.abs(ac[:, None, 1] - bc[None, :, 1])
    return int(np.min(dr + dc))


def _nearest_sprite_index(sprites: list[dict], idx: int) -> int:
    """
    Return the nearest other sprite index.

    Generator guarantees the minimum distance is unique. If an arbitrary external input creates
    a tie, fall back to a visible spatial key based on the candidate bbox.
    """
    src = sprites[idx]
    best_key = None
    best_j = None
    for j, other in enumerate(sprites):
        if j == idx:
            continue
        dist = _sprite_distance(src, other)
        r0, c0, r1, c1 = other["bbox"]
        key = (dist, r0, c0, r1, c1)
        if best_key is None or key < best_key:
            best_key = key
            best_j = j
    assert best_j is not None, "_nearest_sprite_index requires at least two sprites"
    return int(best_j)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow sprite using the border color of its nearest neighboring sprite.

    Input structure:
    - The boundary background color dominates the grid border.
    - Several separated hollow single-color sprites appear on that background, and their enclosed
      interior holes are still background-colored in the input.

    Rule:
    - Detect each sprite and the bg holes enclosed inside it.
    - For every sprite, compare it against all other sprites using minimum Manhattan distance
      between colored cells, then choose the nearest other sprite.
    - Fill every enclosed hole of the sprite with that nearest sprite's border color.

    Key insight: The fill color is no longer encoded in an edge legend; it is transferred from a
    spatially closest peer sprite, so the solver must reason over object distances rather than a
    local border-to-fill lookup table.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    sprites = _extract_sprites(g, bg)
    if len(sprites) < 2:
        return g.copy()

    out = g.copy()
    for i, sprite in enumerate(sprites):
        if not sprite["holes"]:
            continue
        src_idx = _nearest_sprite_index(sprites, i)
        fill = int(sprites[src_idx]["color"])
        for hole in sprite["holes"]:
            for r, c in hole:
                out[int(r), int(c)] = fill
    return out


# --------------- generator ---------------

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_TINY_H_THRESHOLD = 5

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
_MAX_RETRIES = 300


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


def _choose_sprite_count(H: int, W: int, rng: random.Random) -> int:
    """Choose 2-4 sprites, scaling the upper bound with the available area."""
    choices = [2]
    area = H * W
    if H >= 5 and W >= 10 and area >= 45:
        choices.extend([3, 3])
    if H >= 7 and W >= 12 and area >= 84:
        choices.append(4)
    return int(rng.choice(choices))


def _find_valid_positions(
    H: int, W: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements with no overlap/padding violations and with holes off the grid boundary."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            if occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == H - 1 or c0 + hc == W - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    """Render sprite border cells onto the grid."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color


def _global_holes(r0: int, c0: int, holes: list[np.ndarray]) -> list[np.ndarray]:
    """Translate local hole coordinates into grid coordinates."""
    return [np.asarray([(r0 + int(r), c0 + int(c)) for r, c in hpts], dtype=int) for hpts in holes]


def _global_coords(r0: int, c0: int, mask: np.ndarray) -> np.ndarray:
    """Return global colored-cell coordinates for a placed sprite."""
    pts = np.argwhere(mask)
    return np.asarray([(r0 + int(r), c0 + int(c)) for r, c in pts], dtype=int)


def _nearest_is_unique(sprites: list[dict]) -> bool:
    """Reject layouts where any sprite has a tied nearest neighbor."""
    for i, src in enumerate(sprites):
        dists = [_sprite_distance(src, other) for j, other in enumerate(sprites) if j != i]
        dists.sort()
        if len(dists) < 1:
            return False
        if len(dists) > 1 and dists[0] == dists[1]:
            return False
    return True


def _build_output_from_layout(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    """Fill each sprite's holes from its nearest other sprite color."""
    out = inp.copy()
    for i, sprite in enumerate(sprites):
        src_idx = _nearest_sprite_index(sprites, i)
        fill = int(sprites[src_idx]["color"])
        for hole in sprite["holes"]:
            for r, c in hole:
                out[int(r), int(c)] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    total_sprites = _choose_sprite_count(H, W, rng)

    max_sh = min(_SPRITE_H_MAX, H)
    max_sw = min(_SPRITE_W_MAX, W)
    if H <= _TINY_H_THRESHOLD:
        max_sh = 3
    if total_sprites == 3:
        max_sh = min(max_sh, 5)
        max_sw = min(max_sw, 5)
        dim_sum_cap = min(_SPRITE_DIM_SUM_CAP, 9)
    elif total_sprites == 4:
        max_sh = min(max_sh, 4)
        max_sw = min(max_sw, 4)
        dim_sum_cap = min(_SPRITE_DIM_SUM_CAP, 8)
    else:
        dim_sum_cap = _SPRITE_DIM_SUM_CAP

    shapes = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > dim_sum_cap:
            sw = max(3, dim_sum_cap - sh)
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    colors = rng.sample(list(range(1, 10)), total_sprites)
    sprite_specs = list(zip(shapes, colors, strict=False))
    rng.shuffle(sprite_specs)
    sprite_specs.sort(key=lambda item: int(item[0].sum()), reverse=True)

    inp = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    placed: list[dict] = []

    for mask, color in sprite_specs:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(H, W, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(inp, r0, c0, mask, int(color))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)] = True
        placed.append(
            {
                "color": int(color),
                "coords": _global_coords(r0, c0, mask),
                "bbox": (r0, c0, r0 + sh - 1, c0 + sw - 1),
                "holes": _global_holes(r0, c0, holes),
            }
        )

    if not _nearest_is_unique(placed):
        return None

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    output = _build_output_from_layout(inp, placed)

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
