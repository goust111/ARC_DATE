# Variant of dbff022c
# Dimension: D2 — 调色板图例的位置与朝向 (symbol_instructions)
# Change: 调色板为2xN或Nx2色块，位于网格边缘（上/下/左），根据所在边缘确定阅读方向：上边缘从上到下读，下边缘从下到上读，左... → 调色板为分散的颜色对而非连续色块

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the D2 variant
#
# design_analysis:
#   concrete_form:
#     - Each legend entry is now one isolated 2-cell domino on the boundary.
#     - Top/bottom entries are vertical and read from the edge inward.
#     - Left/right entries are horizontal and read from the edge inward.
#     - Multiple entries may appear on multiple edges, so the solver must gather
#       them before building the border->fill table.
#   behavior_trace:
#     old:
#       One contiguous 2xN or Nx2 strip could be read by zipping two aligned lines.
#     new:
#       Example: top has (4,2), right has (6,7), bottom has (9,3). There is no
#       single rectangle covering all entries, so the solver must detect isolated
#       edge dominoes one by one and merge them into a mapping.
#     difference:
#       Legend discovery changes completely; enclosed-hole filling can stay the same.
#   complexity_calibration:
#     - Keep a small legend, usually exactly 3 pairs, so the scatter is readable
#       without overcrowding the grid.
#     - Force the legend to use at least two edges so the scatter is visible and
#       the samples do not collapse back to the original single-strip reading.
#     - Large grids may include one distractor sprite; cramped grids may omit it
#       to preserve a clean, legal layout.
#   implicit_assumption_audit:
#     - Original `_best_palette()` relied on one contiguous edge block. That is no
#       longer true and must be replaced.
#     - Original `_palette_mapping()` assumed one aligned pair of rows/cols. The
#       variant instead assembles the mapping from isolated edge dominoes.
#     - `_fill_enclosed_bg_regions()` only needs a correct mapping and a legend mask,
#       so it remains valid.
#   placement_legality:
#     - Every legend domino is 4-neighbor isolated from all other non-bg cells.
#     - Sprites are kept away from the outer border, so non-bg boundary cells belong
#       only to legend pieces.
#     - Palette pieces and sprites still keep background padding for clean separation.
#   observability:
#     - At least two legend edges are used.
#     - At least two filled sprites use keys coming from different legend edges.
#     - No rule branch depends on color id ordering.
#   new_solver_assumptions:
#     - Any non-bg boundary domino with only bg outside-neighbors is a legend pair.
#     - Border colors are unique legend keys.
#     - `generate_input()` enforces both assumptions.
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


def _rect_with_padding(cells: tuple[tuple[int, int], tuple[int, int]], h: int, w: int) -> tuple[int, int, int, int]:
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
) -> tuple[dict[int, int], np.ndarray, list[tuple[str, int, int, int]]]:
    """
    Returns (mapping border->fill, palette_mask, pair_infos).

    pair_infos entries are (side, idx, border_color, fill_color).
    """
    h, w = g.shape
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}
    pair_infos: list[tuple[str, int, int, int]] = []
    seen: set[frozenset[tuple[int, int]]] = set()

    for side in _EDGE_SIDES:
        for idx in _pair_slots(side, h, w):
            cells = _pair_cells(side, idx, h, w)
            key = frozenset(cells)
            if key in seen:
                continue
            border_cell, fill_cell = cells
            b = int(g[border_cell])
            f = int(g[fill_cell])
            if b == bg or f == bg:
                continue
            if not _isolated_pair(g, bg, cells):
                continue
            if b in mapping:
                assert mapping[b] == f, "legend border color repeated with conflicting fill"
            mapping[b] = f
            pal_mask[border_cell] = True
            pal_mask[fill_cell] = True
            pair_infos.append((side, int(idx), b, f))
            seen.add(key)

    assert mapping, "_collect_palette_pairs must find at least one isolated edge pair"
    return mapping, pal_mask, pair_infos


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
    Fill hollow sprites' interiors by reading scattered color-pair legend pieces on the grid boundary.

    Input structure:
    - The dominant border color is bg, and several hollow single-color sprites sit away from the outer frame.
    - The border->fill legend is not one compact strip anymore: it is split into isolated 2-cell dominoes placed on one or more grid edges, each read from the edge cell toward the interior.

    Rule:
    - Detect every isolated edge domino and convert it into one border-color to fill-color mapping entry.
    - Find enclosed bg regions that are not part of the legend and do not touch the grid boundary.
    - For each such hole, identify its dominant adjacent border color and, if that color is a legend key, fill the hole with the mapped color.

    Key insight: The mapping must be assembled from multiple tiny edge-local clues first; only after collecting those scattered pairs can the usual hole-filling step be applied.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask, _ = _collect_palette_pairs(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)

    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 6, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_TINY_H_THRESHOLD = 5

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 3
_MIN_USED_EDGES = 2

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 500
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


def _choose_colors(rng: random.Random, n: int) -> tuple[list[int], list[int], int, dict[int, int]] | None:
    """Pick palette border/fill colors plus one distractor border color."""
    all_c = list(range(1, 10))
    border_keys = rng.sample(all_c, n)

    for _ in range(100):
        fill_values = rng.sample(all_c, n)
        if all(b != f for b, f in zip(border_keys, fill_values, strict=False)):
            break
    else:
        return None

    distractor_color = rng.choice([c for c in all_c if c not in border_keys])
    mapping = dict(zip(border_keys, fill_values, strict=False))
    return border_keys, fill_values, distractor_color, mapping


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

        edge_target = rng.randint(_MIN_USED_EDGES, min(len(usable_edges), n))
        required_edges = rng.sample(usable_edges, edge_target)

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
        if ok and len({side for side, _, _ in chosen}) >= _MIN_USED_EDGES:
            return chosen
    return None


def _choose_filled_keys(
    rng: random.Random, pair_infos: list[tuple[str, int, int, int]], num_filled: int
) -> list[int] | None:
    """Choose filled-sprite keys, ensuring at least two legend edges contribute."""
    edge_to_keys: dict[str, list[int]] = {}
    for side, _idx, border, _fill in pair_infos:
        edge_to_keys.setdefault(side, []).append(border)

    used_edges = list(edge_to_keys)
    if len(used_edges) < _MIN_USED_EDGES:
        return None

    filled: list[int] = []
    for side in rng.sample(used_edges, _MIN_USED_EDGES):
        filled.append(rng.choice(edge_to_keys[side]))

    remaining = num_filled - len(filled)
    if remaining > 0:
        others = [border for _side, _idx, border, _fill in pair_infos if border not in filled]
        filled.extend(rng.sample(others, remaining))

    rng.shuffle(filled)
    return filled


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]) -> list[tuple[int, int]]:
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
    """Render a sprite onto the grid: border cells in color, hole cells in fill (if given)."""
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


def _palette_capacity(h: int, w: int) -> int:
    """Loose upper bound on isolated pair count under 1-cell same-edge gaps."""
    top_bottom = ((w - 2) + 1) // 2
    left_right = ((h - 2) + 1) // 2
    return 2 * top_bottom + 2 * left_right


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    max_pairs = min(_PALETTE_N_MAX, _palette_capacity(h, w))
    if max_pairs < _PALETTE_N_MIN:
        return None
    n = 3 if max_pairs >= 3 else 2

    max_sh = min(5, h - 2)
    max_sw = min(5, w - 2)
    if h <= _TINY_H_THRESHOLD:
        max_sh = min(max_sh, 3)
    if max_sh < 3 or max_sw < 3:
        return None

    num_filled = 2
    has_distractor = h >= 9 and w >= 12 and rng.random() < 0.5
    total_sprites = num_filled + int(has_distractor)

    shapes = []
    for _ in range(total_sprites):
        mask = _sample_shape(max_sh, max_sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    palette_positions = _select_palette_positions(h, w, n, rng)
    if palette_positions is None:
        return None

    color_info = _choose_colors(rng, n)
    if color_info is None:
        return None
    border_keys, fill_values, distractor_color, mapping = color_info

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

    filled_keys = _choose_filled_keys(rng, pair_infos, num_filled)
    if filled_keys is None:
        return None

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            c, f = filled_keys[i], mapping[filled_keys[i]]
        else:
            c, f = distractor_color, None
        infos.append((mask, c, f))

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
        if fill is not None:
            for hpts in _get_enclosed_holes(mask):
                for hr, hc in hpts:
                    inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    pair_count = len(_collect_palette_pairs(inp, bg)[2])
    if pair_count != n:
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
