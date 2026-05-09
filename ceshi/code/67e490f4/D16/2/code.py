# Variant of 67e490f4
# Dimension: D16 — 多步处理：填充顺序 (multi_step)
# Change: 单步处理——所有洞独立并行填充，填充顺序不影响结果... → 两阶段处理：先匹配形状再查表填色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for the D16 variant
# ----------------------------------
# variant:
#   core_change:
#     original: "洞形状 -> 外部同形精灵的多数颜色 -> 直接填洞"
#     new: "洞形状 -> 外部同形精灵的多数颜色(精灵色) -> 图例查表 -> 最终填充色"
#   concrete_layout:
#     frame:
#       kept: one large monochrome frame with bg-colored holes inside its bbox
#       reason: D16 should layer on top of the old task, not replace its object grammar
#     sprites:
#       kept: small scattered monochrome sprites outside the frame; shape still identifies which hole family they refer to
#     legend:
#       added: one small rectangular ring outside the frame, width fixed at 4
#       structure: border is a dedicated legend color; each interior row is exactly two cells [sprite_color, fill_color]
#       visibility: the legend is a compact but explicit lookup table, easy to spot and read by eye
#   behavior_trace:
#     example:
#       hole_shape: "L triomino"
#       matching_sprites: "three red L sprites, one teal L sprite"
#       stage_1: "majority matched sprite color = red"
#       legend_row: "[red, yellow]"
#       stage_2: "fill every L-shaped hole with yellow, not red"
#     difference_from_original:
#       original would stop after stage_1 and paint red
#       variant must continue to stage_2 and repaint via the legend
#   complexity_calibration:
#     preserved:
#       - rotation/flip matching
#       - plurality over repeated outside sprites
#       - symmetric multi-hole layouts
#     controlled:
#       - one legend only
#       - one mapping row per distinct hole shape color
#       - fill colors are a non-identity permutation of sprite colors, so the second step is always active but not visually chaotic
#   old_assumptions_that_break:
#     - old solver treated every outside non-bg component as a sprite; now the legend must be found and removed before sprite extraction
#     - old generator made output color equal winning sprite color; now output color must be separated from sprite color by a visible mapping
#     - old placement occupancy only respected non-bg pixels; bg holes inside the frame could look free, so the whole frame bbox must be blocked during legend/sprite placement
#   legality_and_spacing:
#     - the legend is placed with the same no-touch padding as sprites
#     - a dedicated occupancy canvas marks the whole frame bbox as occupied, preventing legend/sprite intrusion into holes
#     - the legend border color is excluded from sprite colors, so its border cannot merge with sprites or be mistaken for a winning color
#   observability:
#     - every mapping row is non-identity: left color != right color
#     - output uses the mapped fill colors, so the difference from the original task is visible on every sample
#     - row order can vary, but the legend always provides repeated structure: many holes share a few mapped shape families
#   cleanliness:
#     - the legend is the only extra structure outside the frame besides sprites
#     - it is compact, isolated by bg, and removed from sprite parsing by the solver
#   new_solver_assumptions_and_generator_guarantees:
#     - assumption: exactly one small 4-column legend ring exists outside the frame
#       guarantee: generator creates one legend patch and no other border-ring object
#     - assumption: each hole shape has at least one matching sprite plurality color
#       guarantee: generator places multiple winning sprites per shape and keeps plurality strict
#     - assumption: every winning sprite color appears as a left entry in the legend
#       guarantee: legend rows are generated from the exact per-shape winning colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes in the main frame by shape matching first, then recoloring through an outside legend.

    Input structure:
    - bg is the most common color. One large monochrome frame contains bg-colored holes.
    - Outside the frame: small monochrome sprites whose shapes match the hole families.
    - Also outside the frame: one compact rectangular legend box; each row stores [sprite color -> fill color].

    Rule:
    - Crop to the frame's bounding box and start from a solid frame-colored canvas.
    - For each hole, find the outside sprite shape that matches it up to rotation/flip, then take the most frequent matched sprite color.
    - Convert that sprite color through the legend lookup table and use the mapped color to fill the hole.

    Key insight: The outside sprites no longer directly tell the answer color. Their shape identifies the hole, their majority color selects a legend key, and the legend determines the final paint color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _all_mask_keys(mask: np.ndarray):
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

    def _find_legend(outside_grid: np.ndarray):
        candidates = []
        for comp in find_components(outside_grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
            bb = tuple(map(int, comp["bbox"]))
            sub = crop_bbox(outside_grid, bb)
            bh, bw = map(int, sub.shape)
            border_col = int(comp["color"])
            if bw != 4 or bh < 5:
                continue
            border = np.zeros_like(sub, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if int(comp["area"]) != int(border.sum()):
                continue
            if not np.all(sub[border] == int(border_col)):
                continue
            inner = sub[1:-1, 1:-1]
            if inner.shape[1] != 2 or inner.shape[0] < 3:
                continue
            if np.any(inner == int(bg)) or np.any(inner == int(border_col)):
                continue
            if any(int(inner[rr, 0]) == int(inner[rr, 1]) for rr in range(inner.shape[0])):
                continue
            candidates.append((int(comp["area"]), bb, int(border_col), inner.copy()))
        assert candidates
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0]

    # STEP 1) Find the main template frame: the unique largest monochrome non-bg component.
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = tuple(map(int, frame["bbox"]))
    r0, c0, r1, c1 = fb

    sub = crop_bbox(g, fb)
    th, tw = map(int, sub.shape)

    # STEP 2) Extract hole masks from the frame bbox.
    hole_mask = (sub != int(frame_col))
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append((((hh, ww), m.tobytes()), (int(hr0), int(hc0)), m))

    # STEP 3) In the outside region, locate the legend first and remove it before reading sprites.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    _legend_area, legend_bbox, _legend_col, legend_inner = _find_legend(outside)
    lr0, lc0, lr1, lc1 = legend_bbox
    legend_map = {int(legend_inner[rr, 0]): int(legend_inner[rr, 1]) for rr in range(legend_inner.shape[0])}
    outside[lr0 : lr1 + 1, lc0 : lc1 + 1] = int(bg)

    # STEP 4) Extract remaining outside sprites and tally colors by shape key.
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        for key in _all_mask_keys(m):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 5) Render output: shape match first, legend recolor second.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        sprite_col = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        fill_col = legend_map.get(int(sprite_col))
        assert fill_col is not None
        patch = np.full(hmask.shape, int(fill_col), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering
WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

# Legend
LEGEND_WIDTH = 4

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


def _all_rot_flip(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _build_shape_library():
    raw = [
        [[1]], [[1, 1]], [[1], [1]], [[1, 1, 1]], [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]], [[1, 0], [1, 1]], [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]], [[1, 1], [1, 0]], [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]],
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
    return list({
        (r, c),
        (r, tw - 1 - c - sw + 1),
        (th - 1 - r - sh + 1, c),
        (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
    })


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


def _validate_positions(occ, shape, positions, th, tw):
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
                for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                        if (nr, nc) not in new_cells:
                            return None
    return positions


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
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        s for s in SHAPE_LIBRARY
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


def _random_derangement(items: list[int]) -> list[int]:
    perm = items[:]
    while True:
        random.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return perm[:]


def _assign_shape_colors(n_shapes: int, bg: int, frame_col: int):
    palette = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(palette) < n_shapes + 1:
        return None
    legend_col = random.choice(palette)
    sprite_pool = [c for c in palette if c != legend_col]
    sprite_colors = random.sample(sprite_pool, n_shapes)
    fill_colors = _random_derangement(sprite_colors)
    return legend_col, sprite_colors, fill_colors


def _build_output(occ, hole_placements, frame_col, bg):
    th, tw = occ.shape
    assigned = _assign_shape_colors(len(hole_placements), frame_col=frame_col, bg=bg)
    if assigned is None:
        return None
    legend_col, sprite_colors, fill_colors = assigned

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, pls), sprite_col, fill_col in zip(hole_placements, sprite_colors, fill_colors):
        all_pos = []
        for positions in pls:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, sprite_col, fill_col, all_pos))
    return output, shape_info, legend_col


def _build_legend(shape_info, legend_col):
    rows = [(int(sprite_col), int(fill_col)) for _shape, sprite_col, fill_col, _pos in shape_info]
    random.shuffle(rows)
    legend = np.full((len(rows) + 2, LEGEND_WIDTH), int(legend_col), dtype=int)
    for rr, (sprite_col, fill_col) in enumerate(rows, start=1):
        legend[rr, 1] = int(sprite_col)
        legend[rr, 2] = int(fill_col)
    return legend


def _place_sprite(inp, placement, ori, color, bg):
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _place_legend(inp, placement, legend, bg):
    try:
        r, c = random_free_location(placement, legend, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, legend, r, c, transparent=bg)
    paste(inp, legend, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, placement, shape_info, bg, frame_col, legend_col):
    other_pool = [c for c in range(NUM_COLORS) if c not in {bg, frame_col, legend_col}]
    for shape, sprite_col, _fill_color, _pos in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != int(sprite_col)]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, min(n_other, len(losers)))

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), int(sprite_col), bg):
                return False

        for oc in oc_list:
            n_oc = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), int(oc), bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
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
    output, shape_info, legend_col = result
    legend = _build_legend(shape_info, legend_col)

    fr = random.randint(0, GRID_SIZE - th)
    fc_pos = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    frame_patch = np.where(occ, frame_col, bg).astype(int)
    paste(inp, frame_patch, fr, fc_pos)

    placement = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placement[fr : fr + th, fc_pos : fc_pos + tw] = int(frame_col)

    if not _place_legend(inp, placement, legend, bg):
        return None
    if not _scatter_sprites(inp, placement, shape_info, bg, frame_col, legend_col):
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
