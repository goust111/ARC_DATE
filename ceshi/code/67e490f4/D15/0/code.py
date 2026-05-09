# Variant of 67e490f4
# Dimension: D15 — 网格尺寸与框架占比 (meta)
# Change: 固定 30x30 网格，框架 bbox 7-13行 x 7-20列，框架色密度 0.55-0.92... → 小网格紧凑布局
#
# Design analysis for the D15 variant
# -----------------------------------
# spec:
#   core_rule: shape-coded outside sprites still determine how holes inside the frame are filled
#   grid: 15x15 or 20x20
#   frame:
#     count: exactly 1 monochrome frame component, still the largest non-bg object
#     bbox:
#       15x15 case: about 6-8 rows x 6-9 cols
#       20x20 case: about 7-10 rows x 7-11 cols
#     placement: near the middle so the reduced canvas still leaves a ring of space for sprites
#     density: roughly 0.54-0.86 frame-color occupancy inside the bbox
#   holes:
#     shape_area: 1-3 cells only
#     canonical_shapes: point, domino, line-of-3, L-triomino
#     count: typically 4-10 hole instances, from 2-3 canonical shapes
#     spacing: no hole touches the frame border and holes remain 4-disconnected
#   sprites:
#     role: each outside sprite is a low-resolution copy of one hole shape, up to rotation/flip
#     count: reduced compared with the source task; usually 4-10 total sprites
#     plurality_rule: every shape gets a unique winning color count, so no hidden color-id tie break is needed
#
# a) Concrete behavior and trace
#   Original behavior was "large frame + larger outside field". Here the logic is identical but compressed.
#   Example: a 3-cell L hole inside a 7x8 frame on a 15x15 board is matched by tiny outside L sprites.
#   If the outside has two orange L sprites and one green L sprite, the hole becomes orange.
#   The old generator relied on a large board to make this easy to see; the variant stresses the same
#   reasoning under lower resolution and smaller margins.
#
# b) Complexity calibration
#   The grid is smaller, but the task should not collapse into trivial singletons. Using 2-3 repeated
#   canonical shapes keeps repetition-based evidence visible, while limiting shape area to <=3 makes the
#   low-resolution change visually obvious. The winning color is repeated and the loser count is capped,
#   so the plurality can be inferred from a few examples without ambiguity.
#
# c) Solver-assumption audit
#   - "largest non-bg monochrome component is the frame": still valid because the frame area remains far
#     above each sprite's 1-3 cells, and sprites are separated.
#   - "holes are connected non-frame regions inside the frame bbox": still valid because generation keeps
#     a solid border and 4-disconnected holes.
#   - "matching can use exact mask bytes after rotation/flip": still valid because hole/sprite shapes come
#     from the same small library.
#   - Original code broke plurality ties by smaller color id. That would be an invisible rule here, so the
#     variant changes the assumption: ties are disallowed and asserted against.
#
# d) Placement legality
#   The generator keeps one-cell frame borders, uses symmetric placement retries for hole carving, and
#   scatters sprites with Chebyshev padding=1 outside the frame bbox. This prevents overlaps, merged sprite
#   components, and accidental contact with the frame.
#
# e/f/g) Observability, neatness, and color inference
#   The only new axis is scale: board, frame, holes, and sprite counts are all reduced together. The output
#   remains a cropped frame filled by shape-matched colors, so the relation is still visible in 2-4 examples.
#   Colors are not remapped by any hidden numeric property; they are selected only by repeated outside shape
#   evidence. Visual clutter is reduced by central frame placement and fewer total sprites.

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


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) pairs for every rotation and vertical flip."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _all_mask_keys(mask: np.ndarray):
    for key, _ in _all_rot_flip(mask):
        yield key


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _plurality_color(counts: dict[int, int]) -> int:
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, f"expected a unique plurality, got {counts}"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a compact frame by filling tiny holes from shape-matched micro-sprites.

    Input structure:
    - The dominant color is the background of a 15x15 or 20x20 board.
    - One compact monochrome frame sits inside the board; within its bbox are several 1-3-cell holes.
    - Outside the frame are separated tiny sprites whose shapes match the holes up to rotation/flip.

    Rule:
    - Detect the largest non-background monochrome component and crop to its bounding box.
    - Treat every non-frame-color component inside that crop as a hole shape.
    - For each hole, gather all outside sprites with the same shape under rotation/flip and fill the hole
      with the unique plurality color for that shape.

    Key insight:
    - The outside sprites are low-resolution shape keys: shape tells you which hole to color, and repeated
      sprite color tells you what the fill color must be.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

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

    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    hole_mask = (sub != frame_col)
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((((hh, ww), mask.tobytes()), (int(hr0), int(hc0)), mask))
    assert holes

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    sprite_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert sprite_comps

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in sprite_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        for key in _all_mask_keys(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), frame_col, dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        fill_col = _plurality_color(cmap)
        patch = np.full(hmask.shape, fill_col, dtype=int)
        paste(out, patch, hr0, hc0, mask=hmask)

    return out


# --------------- generator (D15 compact layout) ---------------

NUM_COLORS = 10
GRID_SIZE_OPTIONS = (15, 20)
FRAME_SIZE_RANGES = {
    15: {"h": (6, 8), "w": (6, 9)},
    20: {"h": (7, 10), "w": (7, 11)},
}

SHAPE_COUNT_RANGE = (2, 3)
MIN_SHAPES = 2
SHAPE_MAX_AREA = 3
FRAME_DENSITY_RANGE = (0.54, 0.86)
MIN_TOTAL_HOLES = 4
PLACEMENT_RETRIES = 50
MULT_EVEN_WEIGHTS = ([1, 2], [5, 1])
MULT_ODD_WEIGHTS = ([1, 2], [4, 1])

WINNING_SPRITE_RANGE = (2, 3)
MAX_LOSER_COLORS = 1
LOSER_COUNT = 1
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 500


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1], [1], [1]],
        [[1, 0], [1, 1]],
        [[0, 1], [1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    dedup: list[np.ndarray] = []
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen and int(mask.sum()) <= SHAPE_MAX_AREA:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int):
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


def _validate_positions(occ: np.ndarray, shape: np.ndarray, positions, th: int, tw: int):
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
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                        return None
    return positions


def _try_place_quadrant(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    sh, sw = shape.shape
    r_max = th // 2 - sh
    c_max = tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r = random.randint(1, r_max)
    c = random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
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
    return _validate_positions(occ, shape, list({(r, c), (r, tw - sw - c)}), th, tw)


def _try_place_vcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
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
    return _validate_positions(occ, shape, list({(r, c), (th - sh - r, c)}), th, tw)


def _try_place_center(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _try_place_symmetric(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    placers = [_try_place_quadrant, _try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ: np.ndarray, shape: np.ndarray, positions):
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th: int, tw: int):
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= max_sh and shape.shape[1] <= max_sw and int(shape.sum()) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    available.sort(key=lambda shape: int(shape.sum()), reverse=True)

    occ = np.ones((th, tw), dtype=bool)
    hole_placements: list[tuple[np.ndarray, list[list[tuple[int, int]]]]] = []
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if (th % 2 == 0 or tw % 2 == 0) else MULT_ODD_WEIGHTS

    for shape in available:
        if len(hole_placements) >= k_target:
            break
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placed_groups = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed_groups.append(positions)
                    break
        if placed_groups:
            hole_placements.append((shape, placed_groups))

    if len(hole_placements) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(group) for _, groups in hole_placements for group in groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ: np.ndarray, hole_placements, frame_col: int, bg: int):
    th, tw = occ.shape
    avail_colors = [color for color in range(NUM_COLORS) if color not in (bg, frame_col)]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, groups), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in groups:
            for pr, pc in positions:
                patch = np.full(shape.shape, fill_col, dtype=int)
                paste(output, patch, pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info


def _choose_frame_origin(grid_size: int, th: int, tw: int):
    r_mid = (grid_size - th) // 2
    c_mid = (grid_size - tw) // 2
    r_lo = max(0, r_mid - 1)
    r_hi = min(grid_size - th, r_mid + 1)
    c_lo = max(0, c_mid - 1)
    c_hi = min(grid_size - tw, c_mid + 1)
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _place_sprite(inp: np.ndarray, placement: np.ndarray, ori: np.ndarray, color: int, bg: int):
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info, bg: int, frame_col: int):
    placement = inp.copy()
    other_pool = [color for color in range(NUM_COLORS) if color not in (bg, frame_col)]

    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        loser_candidates = [color for color in other_pool if color != fill_color]
        n_loser_colors = random.randint(0, min(MAX_LOSER_COLORS, len(loser_candidates)))
        loser_colors = random.sample(loser_candidates, n_loser_colors)

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        for loser_color in loser_colors:
            for _ in range(LOSER_COUNT):
                if not _place_sprite(inp, placement, random.choice(orientations), loser_color, bg):
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_size = random.choice(GRID_SIZE_OPTIONS)
    size_cfg = FRAME_SIZE_RANGES[grid_size]
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([color for color in range(NUM_COLORS) if color != bg])
    th = random.randint(*size_cfg["h"])
    tw = random.randint(*size_cfg["w"])

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    result = _build_output(occ, hole_placements, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr, fc = _choose_frame_origin(grid_size, th, tw)
    inp = np.full((grid_size, grid_size), bg, dtype=int)
    frame_patch = np.where(occ, frame_col, bg).astype(int)
    paste(inp, frame_patch, fr, fc)

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
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
