# Combo variant of 67e490f4 (index 7)
# combo_spec: D9/2+D17/0
# Name: 颜色图例+映射填充

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis for the combo variant
# a) Concrete behavior trace:
#    The input still has one large frame with holes plus scattered outside sprites, but now one corner
#    contains a compact kx2 legend. Each row is a visible mapping "sprite color -> fill color".
#    Example: if a 3-cell L-shaped hole matches mostly orange sprites, and the legend row says
#    [orange, teal], then the hole is filled teal rather than orange. The original task ended after
#    majority sprite-color selection; the combo adds a second explicit lookup step.
# b) Complexity calibration:
#    The legend uses only 3-4 rows, and every legend-left color is reused by multiple sprites, so the
#    mapping is easy to notice but still necessary. Sprite colors and fill colors are sampled from
#    disjoint palettes so the task cannot collapse back to "copy the winning sprite color directly".
# c) Hidden assumptions audited from the original solver:
#    - "largest monochrome non-bg component is the frame" breaks once the legend exists, so legend
#      removal must happen first.
#    - "everything outside the frame bbox is a sprite" is false now; the legend occupies that space too.
#    - "winning sprite color is the output color" is replaced by "winning sprite color indexes a legend".
#    - The original tie-break used smaller color id, which is invisible to a human. Here ties are broken
#      by visible legend row order instead.
# d) Placement legality:
#    The generator anchors the legend flush to one corner, keeps the frame at least one bg cell away
#    from it, and reserves both frame bbox and legend bbox when scattering sprites. Sprites also keep
#    Chebyshev padding from each other so components stay separable.
# e) Visual observability:
#    A solver can see three linked structures in every sample: hole shapes, repeated sprite colors, and
#    a corner legend translating those colors. Because the fill colors differ from the sprite colors,
#    the legend is not optional evidence; it is the only visible explanation for the output recoloring.
# f) Cleanliness:
#    The legend is the only dense multi-color block and it is isolated in a corner. The frame stays
#    interior to the board, and sprites are scattered only in the remaining background field.
# g) Change inferability:
#    The color system is deliberately small and repeated. Every possible winning sprite color appears in
#    the legend, and each mapped fill color is used on multiple hole instances inside the symmetric frame.
#
# New solver assumptions introduced by this combo, and how the generator guarantees them:
# - There is exactly one solid kx2 corner legend with no bg gaps.
# - Legend-left colors are unique and cover all sprite colors that are allowed to win.
# - The frame remains the largest monochrome component after legend removal.
# - All hole shapes have at least one matching outside sprite under rotation/flip.
# - Losing sprites, when present, use colors that also exist on the legend-left side, so tie-break and
#   lookup never depend on invisible color-id ordering.


NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 18)
FRAME_BORDER = 1

# Hole layout
SHAPE_COUNT_RANGE = (3, 4)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.90)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering
WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1
SPRITE_BORDER = 1

# Legend
LEGEND_WIDTH = 2
LEGEND_GAP = 1
LEGEND_CORNERS = ("tl", "tr", "bl", "br")

# Generation retry budget
MAX_GEN_ATTEMPTS = 400


def _all_mask_keys(mask: np.ndarray):
    """Yield shape keys for all rotation/flip variants of a boolean mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes())


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Return distinct orientation masks."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


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
    for shape in raw:
        m = np.array(shape, dtype=bool)
        key = _canonical_key(m)
        if key not in seen:
            seen.add(key)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _detect_legend(g: np.ndarray, bg: int):
    """Find the unique corner kx2 legend and decode row order."""
    H, W = g.shape
    legend_candidates = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if ww != LEGEND_WIDTH or hh < MIN_SHAPES:
            continue
        touches_corner = (r0 == 0 or r1 == H - 1) and (c0 == 0 or c1 == W - 1)
        if not touches_corner:
            continue
        sub = crop_bbox(g, (r0, c0, r1, c1))
        if np.any(sub == int(bg)):
            continue
        legend_candidates.append((int(comp["area"]), (r0, c0, r1, c1)))

    assert len(legend_candidates) == 1
    legend_bbox = legend_candidates[0][1]
    legend = crop_bbox(g, legend_bbox)

    legend_map: dict[int, int] = {}
    legend_rank: dict[int, int] = {}
    for rr in range(int(legend.shape[0])):
        sprite_col = int(legend[rr, 0])
        fill_col = int(legend[rr, 1])
        assert sprite_col != bg and fill_col != bg
        assert sprite_col not in legend_map
        legend_map[sprite_col] = fill_col
        legend_rank[sprite_col] = rr
    return legend_bbox, legend_map, legend_rank


def _extract_frame_and_holes(work: np.ndarray, bg: int):
    """Remove nothing else; find the frame and collect hole masks inside its bbox."""
    frame_comps = find_components(work, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert frame_comps
    frame_area = max(int(comp["area"]) for comp in frame_comps)
    frame_candidates = [comp for comp in frame_comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    frame_bbox = tuple(map(int, frame["bbox"]))

    sub = crop_bbox(work, frame_bbox)
    hole_mask = sub != int(frame_col)
    holes = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((((mask.shape), mask.tobytes()), (int(hr0), int(hc0)), mask))
    return frame_col, frame_bbox, holes, sub.shape


def _collect_sprites_by_key(g: np.ndarray, work: np.ndarray, bg: int, frame_bbox):
    """Group outside sprites by canonical hole key and sprite color."""
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = work.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    sprite_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in sprite_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        sprite_col = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[sprite_col] = int(bucket.get(sprite_col, 0)) + 1
    return sprites_by_key


def _render_solution(
    frame_col: int,
    out_shape: tuple[int, int],
    holes,
    sprites_by_key,
    legend_map: dict[int, int] | None,
    legend_rank: dict[int, int] | None,
) -> np.ndarray:
    """Render either the combo solution or the direct-fill baseline."""
    out = np.full(out_shape, int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        if legend_rank is None:
            sprite_col = max(cmap.items(), key=lambda kv: (int(kv[1]), -int(kv[0])))[0]
        else:
            ranked = sorted(cmap.items(), key=lambda kv: (-int(kv[1]), int(legend_rank[int(kv[0])])))
            sprite_col = int(ranked[0][0])
        fill_col = int(sprite_col if legend_map is None else legend_map[sprite_col])
        patch = np.full(hmask.shape, fill_col, dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)
    return out


def _solve_direct_fill_baseline(grid: np.ndarray) -> np.ndarray:
    """Reference the pre-combo behavior on the same input after stripping the legend."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend_bbox, _legend_map, _legend_rank = _detect_legend(g, bg)

    work = g.copy()
    lr0, lc0, lr1, lc1 = map(int, legend_bbox)
    work[lr0 : lr1 + 1, lc0 : lc1 + 1] = int(bg)

    frame_col, frame_bbox, holes, out_shape = _extract_frame_and_holes(work, bg)
    sprites_by_key = _collect_sprites_by_key(g, work, bg, frame_bbox)
    return _render_solution(frame_col, out_shape, holes, sprites_by_key, None, None)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint a holed frame by shape-matching sprites, then translating their colors through a legend.

    Input structure:
    - The background is the most common color. One large monochrome frame contains several hole shapes.
    - Outside the frame are small sprites whose shapes match those holes up to rotation or reflection.
    - One corner contains a compact 2-column legend; each row visibly encodes sprite color -> fill color.

    Rule:
    - Read and remove the corner legend first, then locate the frame and crop to its bounding box.
    - For each hole, gather outside sprites with the same shape and choose the most frequent sprite color.
    - Do not paint with that sprite color directly; instead, look it up in the legend and fill the hole with the mapped color.

    Key insight: Hole shape still chooses which sprite family applies, but sprite color is now only an intermediate key. The final paint comes from an explicit legend-mediated color swap.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    legend_bbox, legend_map, legend_rank = _detect_legend(g, bg)

    work = g.copy()
    lr0, lc0, lr1, lc1 = map(int, legend_bbox)
    work[lr0 : lr1 + 1, lc0 : lc1 + 1] = int(bg)

    frame_col, frame_bbox, holes, out_shape = _extract_frame_and_holes(work, bg)
    sprites_by_key = _collect_sprites_by_key(g, work, bg, frame_bbox)
    return _render_solution(frame_col, out_shape, holes, sprites_by_key, legend_map, legend_rank)


def _legend_bbox(corner: str, rows: int) -> tuple[int, int, int, int]:
    """Return the legend bbox for a chosen corner on the fixed 30x30 grid."""
    h = int(rows)
    w = int(LEGEND_WIDTH)
    if corner == "tl":
        return 0, 0, h - 1, w - 1
    if corner == "tr":
        return 0, GRID_SIZE - w, h - 1, GRID_SIZE - 1
    if corner == "bl":
        return GRID_SIZE - h, 0, GRID_SIZE - 1, w - 1
    if corner == "br":
        return GRID_SIZE - h, GRID_SIZE - w, GRID_SIZE - 1, GRID_SIZE - 1
    raise ValueError(f"unknown corner: {corner}")


def _expand_bbox(bbox: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    """Expand an inclusive bbox and clamp it to the 30x30 grid."""
    r0, c0, r1, c1 = map(int, bbox)
    return (
        max(0, r0 - margin),
        max(0, c0 - margin),
        min(GRID_SIZE - 1, r1 + margin),
        min(GRID_SIZE - 1, c1 + margin),
    )


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return True iff two inclusive bboxes overlap."""
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_frame_location(th: int, tw: int, legend_bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    """Pick a frame location that stays off the edge and away from the legend."""
    if GRID_SIZE - th - FRAME_BORDER < FRAME_BORDER or GRID_SIZE - tw - FRAME_BORDER < FRAME_BORDER:
        return None
    blocked = _expand_bbox(legend_bbox, LEGEND_GAP)
    for _ in range(200):
        fr = random.randint(FRAME_BORDER, GRID_SIZE - th - FRAME_BORDER)
        fc = random.randint(FRAME_BORDER, GRID_SIZE - tw - FRAME_BORDER)
        frame_bbox = (fr, fc, fr + th - 1, fc + tw - 1)
        if not _bboxes_overlap(frame_bbox, blocked):
            return fr, fc
    return None


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4-fold symmetric mirror positions."""
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _validate_positions(occ, shape, positions, th, tw):
    """Check bounds, overlap, and 4-separation from existing holes."""
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
    """Mark hole cells as unavailable in the frame occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place symmetric hole families inside the frame bbox."""
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= max_sh and shape.shape[1] <= max_sw and int(np.sum(shape)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = th % 2 == 0 or tw % 2 == 0
    if even_dims:
        available.sort(key=lambda shape: int(np.sum(shape)))

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

    total_holes = sum(len(pos) for _, placements in hole_placements for pos in placements)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    """Render target output and assign disjoint sprite/fill color pairs per shape."""
    th, tw = occ.shape
    shape_count = len(hole_placements)
    avail_colors = [c for c in range(NUM_COLORS) if c not in (bg, frame_col)]
    if len(avail_colors) < 2 * shape_count:
        return None

    sprite_colors = random.sample(avail_colors, shape_count)
    remaining = [c for c in avail_colors if c not in sprite_colors]
    fill_colors = random.sample(remaining, shape_count)

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, placements), sprite_col, fill_col in zip(hole_placements, sprite_colors, fill_colors):
        all_positions = []
        for positions in placements:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, int(fill_col), dtype=int), pr, pc, mask=shape)
                all_positions.append((pr, pc))
        shape_info.append(
            {
                "shape": shape,
                "sprite_color": int(sprite_col),
                "fill_color": int(fill_col),
                "positions": all_positions,
            }
        )
    return output, shape_info


def _build_legend(shape_info):
    """Build the visible sprite-color -> fill-color legend."""
    rows = [(int(info["sprite_color"]), int(info["fill_color"])) for info in shape_info]
    random.shuffle(rows)
    legend = np.zeros((len(rows), LEGEND_WIDTH), dtype=int)
    for rr, (sprite_col, fill_col) in enumerate(rows):
        legend[rr, 0] = int(sprite_col)
        legend[rr, 1] = int(fill_col)
    return legend


def _reserve_bbox(placement: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    """Mark an inclusive bbox as occupied on the placement grid."""
    r0, c0, r1, c1 = map(int, bbox)
    placement[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _place_sprite(inp, placement, ori, color, bg):
    """Place one sprite on the board and occupancy grid."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = int(color)
    try:
        r, c = random_free_location(
            placement,
            patch,
            transparent=bg,
            border=SPRITE_BORDER,
            padding=SPRITE_PADDING,
        )
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_col, frame_bbox, legend_bbox):
    """Place winning and losing sprites outside both the frame and the legend."""
    placement = np.full_like(inp, bg)
    _reserve_bbox(placement, frame_bbox, frame_col)
    _reserve_bbox(placement, legend_bbox, frame_col)

    legend_sprite_colors = [int(info["sprite_color"]) for info in shape_info]

    for info in shape_info:
        shape = info["shape"]
        sprite_color = int(info["sprite_color"])
        orientations = _all_orientations(shape)

        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in legend_sprite_colors if c != sprite_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        other_colors = random.sample(losers, n_other) if n_other else []

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), sprite_color, bg):
                return False

        for other_color in other_colors:
            n_other_instances = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_other_instances):
                if not _place_sprite(inp, placement, random.choice(orientations), other_color, bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single output-first construction attempt."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    result = _build_output(occ, hole_placements, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    legend_patch = _build_legend(shape_info)
    corner = random.choice(LEGEND_CORNERS)
    legend_bbox = _legend_bbox(corner, int(legend_patch.shape[0]))
    loc = _sample_frame_location(th, tw, legend_bbox)
    if loc is None:
        return None
    fr, fc = loc
    frame_bbox = (fr, fc, fr + th - 1, fc + tw - 1)

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)
    paste(inp, legend_patch, legend_bbox[0], legend_bbox[1])

    if not _scatter_sprites(inp, shape_info, bg, frame_col, frame_bbox, legend_bbox):
        return None

    try:
        solved = main(inp)
        baseline = _solve_direct_fill_baseline(inp)
        if np.array_equal(solved, output) and not np.array_equal(baseline, output):
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
