# Combo variant of dbff022c (index 3)
# combo_spec: D2/1+D10/2
# Name: 分散颜色对调色板 + 干扰条目

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# design_analysis:
#   concrete_form:
#     - The legend is no longer one compact 2xN or Nx2 strip. Each legend entry is
#       an isolated 2-cell edge domino, read from the boundary inward.
#     - Example trace:
#         old:
#           top strip = [(4->2), (6->7), (8->1)] and the solver zips two rows.
#         new:
#           top has (4->2), right has (6->7), bottom has (8->1), left has (9->3).
#           Only sprites with borders 4 and 6 exist; 8 and 9 are fake legend entries.
#           A separate hollow sprite may use border 5 and remain unfilled because 5
#           is not a legend key.
#         difference:
#           The solver must first gather all scattered edge dominoes, then inspect
#           actual enclosed holes to learn which border colors are relevant, then
#           discard unused legend pairs before filling.
#   complexity_calibration:
#     - Keep the number of used fillable border colors small and repeated through
#       whole sprites, usually 2, so humans can match holes to legend keys.
#     - Force at least one unused legend entry and at least two legend edges, so
#       the combo is visually different from both the original contiguous palette
#       and a trivial single-edge scatter.
#     - Keep exactly one unmapped hollow distractor sprite in most samples so the
#       rule "not every hole gets filled" is visible without overcrowding.
#   implicit_assumption_audit:
#     - Original `_best_palette()` assumed a single contiguous rectangle on one edge;
#       that assumption is invalid and removed entirely.
#     - Original `_palette_mapping()` assumed every discovered entry is usable.
#       In the combo, that is false: some collected pairs are fake and must be
#       filtered by checking which border colors actually surround enclosed holes.
#     - `_fill_enclosed_bg_regions()` in the seed only needed a mapping and hole
#       borders. Here the hole analysis is split out explicitly so filtering can
#       happen before filling.
#     - The new solver assumes every real legend piece is an isolated non-bg edge
#       domino and every fillable hole is surrounded by one dominant sprite color.
#       The generator enforces both.
#   placement_legality:
#     - Legend dominoes occupy boundary slots excluding corners and are separated by
#       one-cell padding, so they never merge into longer strips or touch sprites.
#     - Sprites are placed strictly inside the border frame with one-cell padding
#       from each other and from the legend padding mask.
#     - Hollow interiors stay away from the outer frame, so enclosed-hole detection
#       is unambiguous.
#   observability:
#     - Samples always contain more collected legend pairs than actually used keys.
#       This makes distractor entries directly visible.
#     - Used keys are chosen from at least two legend edges, so the scatter is not
#       mistaken for a clipped compact palette.
#     - No step depends on numeric color ordering; only geometry and adjacency matter.
#   visual_tidiness_and_inference:
#     - The border frame remains mostly bg, legend pieces are tiny and isolated, and
#       sprites live in the interior. This keeps the grid readable.
#     - Color count is capped so humans can plausibly infer which edge pairs are real
#       clues and which ones are noise from just a few examples.
# ---------------------------------------------------------------------------


_EDGE_SIDES = ("top", "bottom", "left", "right")


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _pair_cells(side: str, idx: int, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (border_cell, fill_cell) for one inward-reading edge pair."""
    if side == "top":
        return (0, idx), (1, idx)
    if side == "bottom":
        return (h - 1, idx), (h - 2, idx)
    if side == "left":
        return (idx, 0), (idx, 1)
    return (idx, w - 1), (idx, w - 2)


def _pair_slots(side: str, h: int, w: int) -> list[int]:
    """Candidate anchor indices, excluding literal corners."""
    if side in ("top", "bottom"):
        return list(range(1, w - 1))
    return list(range(1, h - 1))


def _rect_with_padding(
    cells: tuple[tuple[int, int], tuple[int, int]], h: int, w: int
) -> tuple[int, int, int, int]:
    rs = [cells[0][0], cells[1][0]]
    cs = [cells[0][1], cells[1][1]]
    r0 = max(0, min(rs) - 1)
    r1 = min(h, max(rs) + 2)
    c0 = max(0, min(cs) - 1)
    c1 = min(w, max(cs) + 2)
    return r0, r1, c0, c1


def _occupy_with_padding(occ: np.ndarray, cells: tuple[tuple[int, int], tuple[int, int]]) -> None:
    h, w = occ.shape
    r0, r1, c0, c1 = _rect_with_padding(cells, h, w)
    occ[r0:r1, c0:c1] = True


def _can_place_pair(occ: np.ndarray, cells: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    return all(not occ[r, c] for r, c in cells)


def _isolated_pair(g: np.ndarray, bg: int, cells: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """True iff both cells are non-bg and every outside 4-neighbor is bg."""
    h, w = g.shape
    pair = {cells[0], cells[1]}
    for r, c in cells:
        if int(g[r, c]) == bg:
            return False
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in pair:
                if int(g[rr, cc]) != bg:
                    return False
    return True


def _collect_palette_pairs(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[str, int, int, int]], np.ndarray]:
    """
    Return (pair_infos, palette_mask).

    pair_infos entries are (side, idx, border_color, fill_color).
    """
    h, w = g.shape
    pal_mask = np.zeros((h, w), dtype=bool)
    pair_infos: list[tuple[str, int, int, int]] = []
    seen_borders: set[int] = set()

    for side in _EDGE_SIDES:
        for idx in _pair_slots(side, h, w):
            border_cell, fill_cell = _pair_cells(side, idx, h, w)
            b = int(g[border_cell])
            f = int(g[fill_cell])
            if b == bg or f == bg:
                continue
            if not _isolated_pair(g, bg, (border_cell, fill_cell)):
                continue
            assert b not in seen_borders, "legend border colors must be unique"
            seen_borders.add(b)
            pal_mask[border_cell] = True
            pal_mask[fill_cell] = True
            pair_infos.append((side, int(idx), b, f))

    assert pair_infos, "_collect_palette_pairs must find at least one isolated edge pair"
    return pair_infos, pal_mask


def _analyze_holes(
    g: np.ndarray, bg: int, pal_mask: np.ndarray
) -> list[tuple[np.ndarray, int]]:
    """
    Return enclosed hole infos as (points, dominant_adjacent_border_color).
    """
    h, w = g.shape
    bg_mask = (g == bg) & (~pal_mask)
    hole_infos: list[tuple[np.ndarray, int]] = []
    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(g[rr, cc])
                    if v != bg:
                        adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border = max(adj_counts.items(), key=lambda kv: kv[1])[0]
        hole_infos.append((pts, int(border)))
    return hole_infos


def _effective_mapping(
    pair_infos: list[tuple[str, int, int, int]], hole_infos: list[tuple[np.ndarray, int]]
) -> dict[int, int]:
    """Keep only legend entries whose border color actually appears around a hole."""
    hole_borders = {border for _pts, border in hole_infos}
    return {border: fill for _side, _idx, border, fill in pair_infos if border in hole_borders}


def _fill_holes(
    out: np.ndarray, hole_infos: list[tuple[np.ndarray, int]], mapping: dict[int, int]
) -> None:
    for pts, border in hole_infos:
        if border not in mapping:
            continue
        fill = int(mapping[border])
        for r, c in pts:
            out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by reading scattered edge color pairs and discarding fake legend entries.

    Input structure:
    - The outer border is mostly bg, and the interior contains several hollow single-color sprites.
    - The border->fill legend is broken into isolated 2-cell dominoes scattered along one or more grid edges, each read from the edge cell inward.
    - Some of those dominoes are distractors: they look like normal legend entries but do not correspond to any sprite border in the scene.

    Rule:
    - Detect every isolated edge domino and collect all candidate border->fill pairs.
    - Find enclosed bg holes away from the boundary, and record the dominant border color around each hole.
    - Filter the collected legend pairs down to the entries whose border color actually appears around some hole.
    - Fill each enclosed hole only if its border color survives that filtering step.

    Key insight: The correct mapping is not "all edge pairs"; it is the subset of scattered pairs supported by real hollow sprites in the interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    pair_infos, pal_mask = _collect_palette_pairs(g, bg)
    hole_infos = _analyze_holes(g, bg, pal_mask)
    mapping = _effective_mapping(pair_infos, hole_infos)
    assert mapping, "main: effective palette mapping must be non-empty"

    out = g.copy()
    _fill_holes(out, hole_infos, mapping)
    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_TINY_H_THRESHOLD = 8

_TOTAL_PAIRS_MIN, _TOTAL_PAIRS_MAX = 3, 4
_USED_PAIR_COUNT = 2
_MIN_USED_EDGES = 2

_SPRITE_H_MAX = 6
_SPRITE_W_MAX = 5
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16

_P_HORIZ_WALL = 0.25
_P_VERT_WALL = 0.25
_P_BORDER_REMOVAL = 0.35
_P_DIAMOND = 0.15

_MAX_RETRIES = 600
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point-arrays for bg components not touching the sprite boundary."""
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
    """Retry shape sampling before failing the whole attempt."""
    for _ in range(40):
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
    rng: random.Random, total_pairs: int
) -> tuple[list[int], list[int], int, dict[int, int]]:
    """Choose disjoint border/fill colors plus one non-legend distractor sprite color."""
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:total_pairs]
    fill_values = all_colors[total_pairs : 2 * total_pairs]
    distractor_sprite_color = all_colors[2 * total_pairs]
    mapping = dict(zip(border_keys, fill_values, strict=False))
    return border_keys, fill_values, distractor_sprite_color, mapping


def _palette_capacity(h: int, w: int) -> int:
    """Loose upper bound on isolated pair count under 1-cell gaps."""
    top_bottom = ((w - 2) + 1) // 2
    left_right = ((h - 2) + 1) // 2
    return 2 * top_bottom + 2 * left_right


def _select_palette_positions(
    h: int, w: int, n: int, rng: random.Random
) -> list[tuple[str, int, tuple[tuple[int, int], tuple[int, int]]]] | None:
    """Choose n isolated edge-domino positions spanning at least two edges."""
    candidates: list[tuple[str, int, tuple[tuple[int, int], tuple[int, int]]]] = []
    for side in _EDGE_SIDES:
        for idx in _pair_slots(side, h, w):
            candidates.append((side, int(idx), _pair_cells(side, idx, h, w)))

    usable_edges = [side for side in _EDGE_SIDES if _pair_slots(side, h, w)]
    for _ in range(200):
        occ = np.zeros((h, w), dtype=bool)
        chosen: list[tuple[str, int, tuple[tuple[int, int], tuple[int, int]]]] = []
        used: set[tuple[str, int]] = set()

        required_edges = rng.sample(usable_edges, _MIN_USED_EDGES)
        ok = True
        for side in required_edges:
            opts = [
                cand
                for cand in candidates
                if cand[0] == side and (cand[0], cand[1]) not in used and _can_place_pair(occ, cand[2])
            ]
            if not opts:
                ok = False
                break
            cand = rng.choice(opts)
            chosen.append(cand)
            used.add((cand[0], cand[1]))
            _occupy_with_padding(occ, cand[2])
        if not ok:
            continue

        while len(chosen) < n:
            opts = [cand for cand in candidates if (cand[0], cand[1]) not in used and _can_place_pair(occ, cand[2])]
            if not opts:
                ok = False
                break
            cand = rng.choice(opts)
            chosen.append(cand)
            used.add((cand[0], cand[1]))
            _occupy_with_padding(occ, cand[2])
        if ok and len({side for side, _idx, _cells in chosen}) >= _MIN_USED_EDGES:
            return chosen
    return None


def _choose_used_keys(
    rng: random.Random, pair_infos: list[tuple[str, int, int, int]]
) -> list[int] | None:
    """Choose the actually used legend keys, forcing them to come from two edges."""
    edge_to_keys: dict[str, list[int]] = {}
    for side, _idx, border, _fill in pair_infos:
        edge_to_keys.setdefault(side, []).append(border)

    if len(edge_to_keys) < _MIN_USED_EDGES:
        return None

    used_keys = [rng.choice(edge_to_keys[side]) for side in rng.sample(list(edge_to_keys), _MIN_USED_EDGES)]
    if len(set(used_keys)) != _USED_PAIR_COUNT:
        return None
    return used_keys


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find inset placements where mask fits without overlap and holes stay interior."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if np.any(occ[r0 : r0 + sh, c0 : c0 + sw] & mask):
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
    """Render a sprite: border cells in color, hole cells in fill if requested."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _occupy_sprite(occ: np.ndarray, r0: int, c0: int, mask: np.ndarray) -> None:
    """Mark sprite cells and their 1-cell padding using the actual shape."""
    h, w = occ.shape
    for rr, cc in zip(*np.where(mask), strict=False):
        r = r0 + int(rr)
        c = c0 + int(cc)
        occ[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    max_pairs = min(_TOTAL_PAIRS_MAX, _palette_capacity(h, w), 4)
    if max_pairs < _TOTAL_PAIRS_MIN:
        return None
    total_pairs = rng.randint(_TOTAL_PAIRS_MIN, max_pairs)
    if total_pairs <= _USED_PAIR_COUNT:
        return None

    max_sh = min(_SPRITE_H_MAX, h - 2)
    max_sw = min(_SPRITE_W_MAX, w - 2)
    if h <= _TINY_H_THRESHOLD:
        max_sh = min(max_sh, 4)
    if max_sh < 3 or max_sw < 3:
        return None

    total_sprites = _USED_PAIR_COUNT + 1
    shapes = []
    for _ in range(total_sprites):
        mask = _sample_shape(max_sh, max_sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    palette_positions = _select_palette_positions(h, w, total_pairs, rng)
    if palette_positions is None:
        return None

    border_keys, fill_values, distractor_sprite_color, mapping = _choose_colors(rng, total_pairs)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    pair_infos: list[tuple[str, int, int, int]] = []

    rng.shuffle(palette_positions)
    for (side, idx, cells), border, fill in zip(palette_positions, border_keys, fill_values, strict=False):
        border_cell, fill_cell = cells
        grid[border_cell] = border
        grid[fill_cell] = fill
        pair_infos.append((side, idx, border, fill))
        _occupy_with_padding(occ, cells)

    used_keys = _choose_used_keys(rng, pair_infos)
    if used_keys is None:
        return None

    infos = []
    for i, mask in enumerate(shapes):
        if i < _USED_PAIR_COUNT:
            color = used_keys[i]
            fill = mapping[color]
        else:
            color = distractor_sprite_color
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
        _occupy_sprite(occ, r0, c0, mask)

    output = grid.copy()
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

    try:
        found_pairs, pal_mask = _collect_palette_pairs(inp, bg)
        hole_infos = _analyze_holes(inp, bg, pal_mask)
        effective = _effective_mapping(found_pairs, hole_infos)
    except Exception:
        return None

    if len(found_pairs) != total_pairs:
        return None
    if len(effective) != _USED_PAIR_COUNT:
        return None
    if len(found_pairs) <= len(effective):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
