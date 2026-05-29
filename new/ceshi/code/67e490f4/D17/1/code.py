# Variant of 67e490f4
# Dimension: D17 — 符号编码：颜色图例系统 (symbol_instructions)
# Change: 无显式图例——精灵颜色直接作为填充色使用，频率即为隐式编码... → 形状-颜色编码表
#
# Design analysis
# - Concrete appearance:
#   The large object is still one monochrome frame with bg-colored holes inside. The output is still the
#   cropped frame bbox with each hole filled. The change is outside the frame: instead of many scattered
#   monochrome sprites, the grid now has a tidy legend strip on the right side made of 3-6 separated
#   rectangular sample blocks. Every sample block uses the same symbol color `legend_fg` to draw a hole
#   shape on top of a different background color. That block background color is the hole fill color.
# - Behavior trace on one shape:
#   Original logic: if an L-shaped hole had outside L sprites in red/red/green, the solver counted colors
#   and filled all L holes red. New logic: there is one L sample block, e.g. blue L drawn on yellow tile;
#   the solver ignores frequency entirely and learns L -> yellow from that explicit example.
# - Complexity calibration:
#   The legend keeps 3-6 repeated mappings per puzzle, all sample blocks share one visible symbol color,
#   and fill colors stay distinct. This is enough for a human to notice "same symbol color, varying tile
#   backgrounds" without introducing extra freedom or ambiguous multi-step decoding.
# - Hidden assumptions from the original main() that no longer hold:
#   `sprites_by_key` assumed each outside component's own color was the answer color, and that plurality
#   over many sprites mattered. That is false here because each sample block is deliberately multicolor:
#   one color is only the symbol ink, the other is the actual answer. Also, orientation can differ between
#   hole and legend symbol, so the solver must canonicalize shapes instead of using raw mask bytes.
# - Placement legality:
#   The frame remains the largest monochrome non-bg component. Legend blocks are multicolor components,
#   placed only in a reserved right-side band with bg gaps between blocks and at least two bg columns
#   between the frame bbox and the legend band, so they cannot merge with the frame or each other.
# - Visual observability:
#   The right-side aligned blocks make the new mechanism obvious in 2-3 examples. Because the same
#   `legend_fg` repeats across every sample while only tile backgrounds vary, the viewer can infer that
#   the shape acts as a symbol and the surrounding tile color is the payload. No color-id ordering or
#   invisible metadata is used.
# - New solver assumptions and how the generator guarantees them:
#   1. Every outside informative component is a 2-color legend block.
#      The generator creates only those components outside the frame.
#   2. Exactly one non-bg color is shared across all legend blocks.
#      The generator uses one global `legend_fg` distinct from bg/frame/fills.
#   3. Each hole canonical shape appears in exactly one legend block.
#      The generator builds one sample block per distinct canonical hole shape.
#   4. Fill colors are distinct and never equal to bg/frame/legend colors.
#      The generator samples them from the remaining palette.

from __future__ import annotations

import random
import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
)


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Canonical representative across rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside a large template frame by reading a side legend of shape-color samples.

    Input structure:
    - bg is the most common color. One large monochrome frame object contains bg-colored holes.
    - Outside the frame, there are several separated rectangular sample blocks arranged on one side.
    - Every sample block uses one shared symbol color to draw a hole shape on a colored tile background.

    Rule:
    - Crop to the frame's bounding box and begin with a solid frame-colored canvas.
    - Read each sample block as a mapping from shape -> tile background color, allowing rotation/flip when matching shapes.
    - Fill every frame hole with the color assigned to its matching legend shape.

    Key insight: The outside objects are not answer-colored sprites anymore; they are explicit dictionary entries where the repeated symbol color marks the shape token and the block's background carries the fill color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: largest connected monochrome non-bg component.
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    # STEP 2) Extract holes inside the frame bbox and canonicalize their shapes.
    hole_mask = (sub != int(frame_col))
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append((_canonical_key(m), (int(hr0), int(hc0)), m))

    # STEP 3) Read the outside legend blocks.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    legend_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert legend_comps

    color_sets = []
    for comp in legend_comps:
        coords = comp["coords"]
        vals = set(int(v) for v in g[coords[:, 0], coords[:, 1]])
        assert len(vals) == 2
        color_sets.append(vals)

    shared = set.intersection(*color_sets)
    assert len(shared) == 1
    legend_fg = next(iter(shared))

    legend_map: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in legend_comps:
        coords = comp["coords"]
        vals = g[coords[:, 0], coords[:, 1]]
        fill_colors = sorted({int(v) for v in vals if int(v) != legend_fg})
        assert len(fill_colors) == 1
        fill_color = fill_colors[0]

        sym_coords = coords[vals == legend_fg]
        sr0, sc0, sr1, sc1 = bbox_coords(sym_coords)
        sh = int(sr1 - sr0 + 1)
        sw = int(sc1 - sc0 + 1)
        rel = sym_coords - np.array([sr0, sc0])
        mask = np.zeros((sh, sw), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        key = _canonical_key(mask)

        if key in legend_map:
            assert legend_map[key] == fill_color
        legend_map[key] = fill_color

    # STEP 4) Render output from the learned legend mapping.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        fill_color = legend_map.get(key)
        assert fill_color is not None
        patch = np.full(hmask.shape, int(fill_color), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 13)

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Legend construction
LEGEND_MARGIN = 1
LEGEND_GAP = 1
LEGEND_SIDE_GAP = 2
MAX_LEGEND_COLS = 2

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 0]],
        [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
    ]
    seen, dedup = set(), []
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4-fold mirrored positions (deduplicated)."""
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap, and 4-disconnected from other holes."""
    sh, sw = shape.shape
    new_cells = set()
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    new_cells.add((pr + dr, pc + dc))

    for pr, pc in positions:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if not shape[dr, dc]:
                    continue
                rr, cc = pr + dr, pc + dc
                if not occ[rr, cc]:
                    return None
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                        if (nr, nc) not in new_cells:
                            return None
    return positions


def _try_place_quadrant(occ, shape, th, tw):
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    if th % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    if r < 1 or r + sh > th - 1:
        return None
    c_max = tw // 2 - sw
    if c_max < 1:
        return None
    c = random.randint(1, c_max)
    c_m = tw - 1 - c - sw + 1
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ, shape, th, tw):
    if tw % 2 == 0:
        return None
    sh, sw = shape.shape
    c = tw // 2 - sw // 2
    if c < 1 or c + sw > tw - 1:
        return None
    r_max = th // 2 - sh
    if r_max < 1:
        return None
    r = random.randint(1, r_max)
    r_m = th - 1 - r - sh + 1
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ, shape, th, tw):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _try_place_symmetric(occ, shape, th, tw):
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place holes in the frame interior. Returns (occ, hole_placements) or None."""
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and int(np.sum(s)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = (th % 2 == 0 or tw % 2 == 0)
    if even_dims:
        available.sort(key=lambda s: int(np.sum(s)))

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []

    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS
    for shape in available:
        if len(hole_placements) >= k_target:
            break
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placed = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed.append(positions)
                    break
        if placed:
            hole_placements.append((shape, placed))

    if len(hole_placements) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(pos) for _, pls in hole_placements for pos in pls)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _assign_shape_colors(hole_placements, blocked_colors):
    """Assign one distinct fill color to each canonical hole shape."""
    blocked = set(int(c) for c in blocked_colors)
    avail_colors = [c for c in range(NUM_COLORS) if c not in blocked]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    shape_info = []
    for (shape, pls), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in pls:
            for pr, pc in positions:
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return shape_info


def _render_output(occ, shape_info, frame_col):
    th, tw = occ.shape
    output = np.full((th, tw), int(frame_col), dtype=int)
    for shape, fill_col, positions in shape_info:
        patch = np.full(shape.shape, int(fill_col), dtype=int)
        for pr, pc in positions:
            paste(output, patch, pr, pc, mask=shape)
    return output


def _build_legend_tile(shape, fill_color, legend_fg):
    """One sample block: shape in legend_fg over a fill-colored tile."""
    ori = random.choice(_all_orientations(shape))
    h, w = ori.shape
    tile = np.full((h + 2 * LEGEND_MARGIN, w + 2 * LEGEND_MARGIN), int(fill_color), dtype=int)
    patch = np.full(ori.shape, int(legend_fg), dtype=int)
    paste(tile, patch, LEGEND_MARGIN, LEGEND_MARGIN, mask=ori)
    return tile


def _legend_layout_dims(tiles):
    """Choose the narrowest 1- or 2-column layout that fits vertically."""
    max_h = max(int(tile.shape[0]) for tile in tiles)
    max_w = max(int(tile.shape[1]) for tile in tiles)
    n = len(tiles)
    best = None
    for n_cols in range(1, min(MAX_LEGEND_COLS, n) + 1):
        rows = (n + n_cols - 1) // n_cols
        total_h = rows * max_h + (rows - 1) * LEGEND_GAP
        total_w = n_cols * max_w + (n_cols - 1) * LEGEND_GAP
        if total_h > GRID_SIZE:
            continue
        candidate = (total_w, total_h, n_cols, max_h, max_w)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def _place_legends(inp, tiles, start_col):
    dims = _legend_layout_dims(tiles)
    if dims is None:
        return False
    total_w, total_h, n_cols, max_h, max_w = dims
    band_w = GRID_SIZE - start_col
    if total_w > band_w:
        return False

    top = random.randint(0, GRID_SIZE - total_h)
    left = random.randint(0, band_w - total_w)
    order = list(range(len(tiles)))
    random.shuffle(order)

    for slot_idx, tile_idx in enumerate(order):
        row = slot_idx // n_cols
        col = slot_idx % n_cols
        slot_r = top + row * (max_h + LEGEND_GAP)
        slot_c = left + col * (max_w + LEGEND_GAP)
        tile = tiles[tile_idx]
        tr, tc = tile.shape
        r = slot_r + (max_h - tr) // 2
        c = start_col + slot_c + (max_w - tc) // 2
        paste(inp, tile, r, c)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build frame, then add an explicit right-side legend."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    legend_fg = random.choice([c for c in range(NUM_COLORS) if c not in {bg, frame_col}])

    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)
    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    shape_info = _assign_shape_colors(hole_placements, blocked_colors={bg, frame_col, legend_fg})
    if shape_info is None:
        return None
    output = _render_output(occ, shape_info, frame_col)

    tiles = [_build_legend_tile(shape, fill_color, legend_fg) for shape, fill_color, _ in shape_info]
    dims = _legend_layout_dims(tiles)
    if dims is None:
        return None
    legend_w = dims[0]

    max_fc = GRID_SIZE - tw - LEGEND_SIDE_GAP - legend_w
    if max_fc < 0:
        return None
    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, max_fc)

    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    legend_start = fc + tw + LEGEND_SIDE_GAP
    if not _place_legends(inp, tiles, legend_start):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
