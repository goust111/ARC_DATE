# Combo variant of dbff022c (index 24)
# combo_spec: D2/1+D3/1+D12/0
# Name: 分散颜色对 + 标记像素指示方向 + 分隔调色板

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# design_analysis:
#   concrete_form:
#     - The old compact 2xN / Nx2 legend is replaced by multiple scattered
#       edge-local "micro-legends".
#     - Each micro-legend is a 3-cell straight segment anchored on the border:
#         [marker, border, fill] or [fill, border, marker]
#       when read from the boundary inward.
#     - The repeated marker color is the only repeated palette color and tells
#       the reading direction inside each segment. The middle cell is always the
#       border-key; the far non-marker end is the fill color.
#     - At least two segments are separated by background gaps on the same edge,
#       so the separator structure is visible rather than implied only by sparse
#       placement across different edges.
#   behavior_trace:
#     old:
#       A solver found one contiguous top/bottom/left/right strip, then zipped
#       two aligned lines into border->fill pairs.
#     new:
#       Example on a top edge: column 2 shows [8, 4, 1] and column 6 shows
#       [3, 7, 8], where 8 is the repeated marker color. The first segment means
#       4->1, while the second means 7->3 because the marker moved to the inner
#       end. A third segment may sit on the right edge, e.g. [8, 6, 2].
#     difference:
#       The solver must first gather isolated edge segments, discover the shared
#       marker color, decode direction entry by entry, and only then fill holes.
#   complexity_calibration:
#     - Use exactly 3 legend entries: enough to show scatter, separators, and a
#       repeated marker without overcrowding the grid.
#     - Force the legend to span exactly 2 edges with a 2+1 distribution, so
#       both "scattered" and "separated" are visible in the same sample.
#     - Force both marker directions to appear in every sample; otherwise D3 can
#       collapse back to a fixed edge-reading heuristic.
#     - Keep colors structured: 1 repeated marker color, 3 unique border keys,
#       3 unique fill colors, and 1 distractor sprite color.
#   implicit_assumption_audit:
#     - Original `_best_palette()` assumed one contiguous rectangular palette.
#       This is no longer valid because the palette is split into independent
#       segments on multiple edges.
#     - Original `_palette_mapping()` assumed border/fill order followed the edge
#       side. That also fails: two segments on the same edge can point in opposite
#       directions.
#     - Original logic also assumed every palette cell was part of one dense
#       block. Here separator gaps are background and must be ignored.
#     - `_fill_enclosed_bg_regions()` still only needs a correct mapping and a
#       mask of palette cells, so it remains unchanged.
#   placement_legality:
#     - Palette segments are 4-neighbor isolated from sprites and from each other
#       via 1-cell padding.
#     - Sprites are placed strictly inside the frame, so every non-bg boundary
#       cell belongs to a palette segment.
#     - Entry segments use depth 3, so the generator raises the minimum grid
#       height enough to preserve room for interior sprites.
#   observability:
#     - Every sample contains one edge with two separated segments and another
#       edge with one extra segment.
#     - Both marker directions appear, and the two filled sprites use entries
#       from different edges with opposite marker directions.
#     - The repeated marker color is visually conspicuous and never reused by
#       sprites or fill values, so humans can spot it across 2-4 examples.
#   visual_cleanliness:
#     - The legend uses only thin straight segments with bg separators rather
#       than overlapping blocks, which keeps the border readable.
#     - Sprite placement keeps at least one background cell around each object,
#       avoiding clutter between border glyphs and hollow shapes.
#   inferability:
#     - The color change is not arbitrary: each changed sprite has a boundary
#       color that reappears as the middle cell of exactly one marked edge
#       segment, and the inserted fill comes from the segment's opposite end.
#     - Because both marker directions occur in each sample, the solver cannot
#       succeed by relying on color ordering or edge side alone.
#   new_solver_assumptions:
#     - Every valid legend entry is an isolated non-bg edge segment of length 3.
#     - Exactly one palette color repeats across entries: the marker color.
#     - In each entry, the marker appears on exactly one end, never in the middle.
#     - Border-key colors are unique. The generator enforces all of these.
# ---------------------------------------------------------------------------


_EDGE_SIDES = ("top", "bottom", "left", "right")


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _entry_cells(side: str, idx: int, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return the 3 cells of an edge micro-legend from boundary inward."""
    if side == "top":
        return (0, idx), (1, idx), (2, idx)
    if side == "bottom":
        return (h - 1, idx), (h - 2, idx), (h - 3, idx)
    if side == "left":
        return (idx, 0), (idx, 1), (idx, 2)
    return (idx, w - 1), (idx, w - 2), (idx, w - 3)


def _entry_slots(side: str, h: int, w: int) -> list[int]:
    """Candidate anchor indices, excluding literal corners."""
    if side in ("top", "bottom"):
        if h < 3:
            return []
        return list(range(1, w - 1))
    if w < 3:
        return []
    return list(range(1, h - 1))


def _rect_with_padding(cells: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = max(0, min(rs) - 1)
    r1 = min(h, max(rs) + 2)
    c0 = max(0, min(cs) - 1)
    c1 = min(w, max(cs) + 2)
    return r0, r1, c0, c1


def _occupy_with_padding(occ: np.ndarray, cells: tuple[tuple[int, int], ...]) -> None:
    h, w = occ.shape
    r0, r1, c0, c1 = _rect_with_padding(cells, h, w)
    occ[r0:r1, c0:c1] = True


def _can_place_entry(occ: np.ndarray, cells: tuple[tuple[int, int], ...]) -> bool:
    return all(not occ[r, c] for r, c in cells)


def _isolated_entry(g: np.ndarray, bg: int, cells: tuple[tuple[int, int], ...]) -> bool:
    """True iff the 3 cells are non-bg and every outside 4-neighbor is bg."""
    h, w = g.shape
    cell_set = set(cells)
    for r, c in cells:
        if int(g[r, c]) == bg:
            return False
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cell_set:
                if int(g[rr, cc]) != bg:
                    return False
    return True


def _collect_palette_entries(
    g: np.ndarray, bg: int
) -> tuple[dict[int, int], np.ndarray, list[tuple[str, int, bool, int, int]], int]:
    """
    Returns (mapping border->fill, palette_mask, entry_infos, marker_color).

    entry_infos entries are (side, idx, marker_at_boundary, border_color, fill_color).
    """
    h, w = g.shape
    raw_entries: list[
        tuple[str, int, tuple[tuple[int, int], tuple[int, int], tuple[int, int]], tuple[int, int, int]]
    ] = []
    for side in _EDGE_SIDES:
        for idx in _entry_slots(side, h, w):
            cells = _entry_cells(side, idx, h, w)
            if not _isolated_entry(g, bg, cells):
                continue
            colors = tuple(int(g[r, c]) for r, c in cells)
            raw_entries.append((side, int(idx), cells, colors))

    assert raw_entries, "_collect_palette_entries must find isolated edge segments"

    counts: dict[int, int] = {}
    for _side, _idx, _cells, colors in raw_entries:
        for color in colors:
            counts[color] = counts.get(color, 0) + 1

    marker_candidates = [color for color, cnt in counts.items() if cnt >= 2]
    assert len(marker_candidates) == 1, "expected exactly one repeated marker color"
    marker_color = int(marker_candidates[0])

    mapping: dict[int, int] = {}
    pal_mask = np.zeros((h, w), dtype=bool)
    entry_infos: list[tuple[str, int, bool, int, int]] = []

    for side, idx, cells, colors in raw_entries:
        first, middle, last = colors
        marker_at_boundary = first == marker_color
        marker_at_inner = last == marker_color
        assert marker_at_boundary ^ marker_at_inner, "marker must occupy exactly one end"
        assert middle != marker_color, "marker cannot be in the middle"

        border = int(middle)
        fill = int(last if marker_at_boundary else first)
        if border in mapping:
            assert mapping[border] == fill, "legend border color repeated with conflicting fill"
        mapping[border] = fill
        for r, c in cells:
            pal_mask[r, c] = True
        entry_infos.append((side, idx, bool(marker_at_boundary), border, fill))

    assert mapping, "_collect_palette_entries must decode a non-empty mapping"
    return mapping, pal_mask, entry_infos, marker_color


def _fill_enclosed_bg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
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
    Fill hollow sprites by decoding scattered marked palette segments on the grid boundary.

    Input structure:
    - The dominant border color is bg, and several hollow single-color sprites sit away from the frame.
    - The border->fill legend is split into isolated 3-cell edge segments rather than one compact strip.
    - Each segment contains the same special marker color on exactly one end, a border-key color in the middle,
      and the corresponding fill color on the opposite end. Some edges may contain multiple segments separated
      by background gaps.

    Rule:
    - Detect every isolated edge segment and identify the shared marker color by finding the only repeated legend color.
    - For each segment, use the middle cell as the border key and the non-marker end as its fill value.
    - Build the full border->fill mapping from all segments, then fill each enclosed background hole whose dominant
      surrounding border color appears in that mapping.

    Key insight:
    - The solver cannot rely on one continuous palette or on edge side to determine reading order. It must first
      gather separated edge clues, infer the marker color shared across them, decode each segment's direction, and
      only then apply the original hole-filling step.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask, _entry_infos, _marker = _collect_palette_entries(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

_GRID_H_MIN, _GRID_H_MAX = 8, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15

_PAIR_COUNT = 3
_MIN_USED_EDGES = 2

_SPRITE_H_MAX = 5
_SPRITE_W_MAX = 5
_SPRITE_DIM_SUM_CAP = 9
_SPRITE_CELL_MAX = 16

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.35
_P_DIAMOND = 0.2

_MAX_RETRIES = 600


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
    """Create hollow sprite (h x w) with at least 1 enclosed bg hole, cell count <= cap."""
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
    rng: random.Random,
) -> tuple[int, list[int], list[int], int, dict[int, int]]:
    """Pick one repeated marker color, 3 border keys, 3 fill values, and 1 distractor color."""
    colors = rng.sample(list(range(1, 10)), 8)
    marker_color = colors[0]
    border_keys = colors[1:1 + _PAIR_COUNT]
    fill_values = colors[1 + _PAIR_COUNT : 1 + 2 * _PAIR_COUNT]
    distractor_color = colors[-1]
    mapping = dict(zip(border_keys, fill_values, strict=False))
    return marker_color, border_keys, fill_values, distractor_color, mapping


def _select_palette_entries(
    h: int, w: int, rng: random.Random
) -> tuple[list[tuple[str, int, bool]], tuple[int, int]] | None:
    """
    Choose 3 entries using exactly 2 edges.

    Returns (entries, filled_entry_indices), where each entry is
    (side, idx, marker_at_boundary).
    """
    primary_sides = [side for side in _EDGE_SIDES if len(_entry_slots(side, h, w)) >= 2]
    secondary_sides = [side for side in _EDGE_SIDES if len(_entry_slots(side, h, w)) >= 1]
    if not primary_sides:
        return None

    for _ in range(200):
        primary = rng.choice(primary_sides)
        secondary_choices = [side for side in secondary_sides if side != primary]
        if not secondary_choices:
            return None
        secondary = rng.choice(secondary_choices)

        occ = np.zeros((h, w), dtype=bool)
        entries: list[tuple[str, int, bool]] = []

        for marker_at_boundary in (True, False):
            opts = []
            for idx in _entry_slots(primary, h, w):
                cells = _entry_cells(primary, idx, h, w)
                if _can_place_entry(occ, cells):
                    opts.append(idx)
            if not opts:
                entries = []
                break
            idx = rng.choice(opts)
            cells = _entry_cells(primary, idx, h, w)
            entries.append((primary, int(idx), marker_at_boundary))
            _occupy_with_padding(occ, cells)
        if len(entries) != 2:
            continue

        secondary_marker_at_boundary = rng.random() < 0.5
        opts = []
        for idx in _entry_slots(secondary, h, w):
            cells = _entry_cells(secondary, idx, h, w)
            if _can_place_entry(occ, cells):
                opts.append(idx)
        if not opts:
            continue
        idx = rng.choice(opts)
        entries.append((secondary, int(idx), secondary_marker_at_boundary))

        # Use two entries from different edges and opposite directions.
        filled_entry_indices = (1, 2) if secondary_marker_at_boundary else (0, 2)
        return entries, filled_entry_indices
    return None


def _paint_palette_entries(
    grid: np.ndarray,
    occ: np.ndarray,
    entries: list[tuple[str, int, bool]],
    marker_color: int,
    border_keys: list[int],
    fill_values: list[int],
) -> list[tuple[str, int, bool, int, int]]:
    """Render all palette entries and return their decoded info."""
    infos: list[tuple[str, int, bool, int, int]] = []
    h, w = grid.shape
    for (side, idx, marker_at_boundary), border, fill in zip(entries, border_keys, fill_values, strict=False):
        cells = _entry_cells(side, idx, h, w)
        colors = (marker_color, border, fill) if marker_at_boundary else (fill, border, marker_color)
        for (r, c), color in zip(cells, colors, strict=False):
            grid[r, c] = color
        _occupy_with_padding(occ, cells)
        infos.append((side, idx, marker_at_boundary, border, fill))
    return infos


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


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    palette_choice = _select_palette_entries(h, w, rng)
    if palette_choice is None:
        return None
    entries, filled_entry_indices = palette_choice

    max_sh = min(_SPRITE_H_MAX, h - 2)
    max_sw = min(_SPRITE_W_MAX, w - 2)
    if max_sh < 3 or max_sw < 3:
        return None

    shapes = []
    for _ in range(3):
        mask = _sample_shape(max_sh, max_sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    marker_color, border_keys, fill_values, distractor_color, mapping = _choose_colors(rng)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    pair_infos = _paint_palette_entries(grid, occ, entries, marker_color, border_keys, fill_values)

    filled_keys = [pair_infos[i][3] for i in filled_entry_indices]
    filled_edges = {pair_infos[i][0] for i in filled_entry_indices}
    filled_dirs = {pair_infos[i][2] for i in filled_entry_indices}
    if len(filled_edges) != _MIN_USED_EDGES or len(filled_dirs) != 2:
        return None

    infos = [
        (shapes[0], filled_keys[0], mapping[filled_keys[0]]),
        (shapes[1], filled_keys[1], mapping[filled_keys[1]]),
        (shapes[2], distractor_color, None),
    ]

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
        decoded_mapping, _pal_mask, decoded_infos, decoded_marker = _collect_palette_entries(inp, bg)
        if decoded_marker != marker_color:
            return None
        if len(decoded_infos) != _PAIR_COUNT:
            return None
        if len({side for side, _idx, _dir, _border, _fill in decoded_infos}) != _MIN_USED_EDGES:
            return None
        if len({marker_at_boundary for _side, _idx, marker_at_boundary, _border, _fill in decoded_infos}) != 2:
            return None
        if decoded_mapping != mapping:
            return None
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
