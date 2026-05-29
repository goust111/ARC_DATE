# Combo variant of 67e490f4 (index 19)
# combo_spec: D6/3+D15/0
# Name: 小网格+嵌套框架
#
# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace
#    The source task had one large filled frame whose bg holes were all handled in one flat pass.
#    This combo compresses the board to 15x15 or 20x20 and inserts a genuine frame-within-frame:
#    an outer filled rectangle contains a bg moat, and inside that moat sits a smaller child frame
#    with its own holes. Example: on a 15x15 board, a 10x10 outer frame may contain two outer hole
#    shapes plus a centered 4x4 inner frame with one extra hole shape. Outside the outer bbox sit
#    tiny sprites for all canonical shapes. The output is cropped to the outer bbox, but unlike the
#    source task it is not a single solid color block: the outer frame, the bg moat, the inner frame,
#    and both levels of filled holes all remain visible.
#
# b) Complexity calibration
#    The nested structure must be obvious without becoming noisy on a small board, so the generator
#    uses only tiny shapes (area <= 3), 2-4 distinct canonical shapes total, and a single child
#    frame. Outer-level and inner-level shape inventories are disjoint; otherwise repeated shapes
#    across levels would force an invisible disambiguation rule and weaken the hierarchy cue.
#
# c) Solver-assumption audit
#    The original solver relied on "everything inside the largest frame bbox that is not frame color
#    is a fillable hole". That breaks here because the bg moat around the child frame is not a hole,
#    and the child frame itself must survive into the output. The new solver still assumes the
#    largest monochrome non-bg component is the outer frame, but then it re-detects all monochrome
#    frame components inside that crop and only fills bg components whose entire 4-neighbor boundary
#    is the owning frame color. This excludes the moat automatically because its boundary touches both
#    outer-frame cells and child-frame cells.
#
# d) Placement legality
#    Generation is output-first. The outer frame is a filled rectangle with a carved moat rectangle;
#    the inner frame is placed strictly inside the moat. Hole placement uses occupancy checks so hole
#    cells never touch the bbox border, the moat, or each other in 4-neighborhood. Sprite placement
#    treats the whole outer bbox as blocked, so all sprites remain strictly outside the template area
#    with Chebyshev padding 1.
#
# e) Visual observability
#    A human can see the new rule from a few examples because the child frame is explicit, the output
#    preserves the bg moat, and the same tiny sprite-matching logic appears at both levels. Color is
#    still inferred only from repeated sprite counts; no color-id ordering is used.
#
# f) Visual neatness
#    The outer frame is placed near the center to leave a clean ring for sprites. There is exactly one
#    child frame, one moat, and a small number of hole shapes, which keeps the compact board readable.
#
# g) Inference of color change
#    Every canonical shape gets a unique plurality color among its outside sprites. Loser colors are
#    capped at one per shape, so the fill rule stays easy to infer even under the reduced resolution.

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


NUM_COLORS = 10
GRID_SIZE_OPTIONS = (15, 20)
MOAT_GAP = 1
SPRITE_PADDING = 1
PLACEMENT_RETRIES = 80
MAX_GEN_ATTEMPTS = 800
_GRID_SIZE_CURSOR = 0

INNER_SIZE_RANGES = {
    15: {"h": (3, 3), "w": (3, 3), "shell": (3, 3)},
    20: {"h": (4, 5), "w": (4, 5), "shell": (2, 3)},
}

LEVEL_CONFIG = {
    15: {
        "outer_shape_count": (1, 1),
        "inner_shape_count": (1, 1),
        "outer_mult": (2, 3),
        "inner_mult": (1, 1),
        "outer_min_holes": 2,
        "inner_min_holes": 1,
    },
    20: {
        "outer_shape_count": (2, 2),
        "inner_shape_count": (1, 2),
        "outer_mult": (1, 2),
        "inner_mult": (1, 2),
        "outer_min_holes": 2,
        "inner_min_holes": 1,
    },
}
SHAPE_MAX_AREA = 3
WINNING_SPRITE_RANGE = (2, 3)
LOSER_SPRITE_COUNT = 1


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) pairs for all rotations and vertical flips."""
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
        if int(mask.sum()) > SHAPE_MAX_AREA:
            continue
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _coords_to_mask(coords: np.ndarray):
    r0, c0, r1, c1 = bbox_coords(coords)
    hh = int(r1 - r0 + 1)
    ww = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((hh, ww), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return (int(r0), int(c0)), mask


def _extract_owned_holes(frame_crop: np.ndarray, frame_col: int, bg: int):
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    bg_mask = frame_crop == bg
    fh, fw = frame_crop.shape
    for coords in components_mask(bg_mask, connectivity=4):
        comp = np.zeros((fh, fw), dtype=bool)
        comp[coords[:, 0], coords[:, 1]] = True
        owned = True
        for rr, cc in coords:
            if rr == 0 or rr == fh - 1 or cc == 0 or cc == fw - 1:
                owned = False
                break
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not comp[nr, nc] and int(frame_crop[nr, nc]) != int(frame_col):
                    owned = False
                    break
            if not owned:
                break
        if not owned:
            continue
        (r0, c0), mask = _coords_to_mask(coords)
        holes.append(((mask.shape, mask.tobytes()), (r0, c0), mask))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a compact nested template by filling hole shapes at two frame levels from outside sprites.

    Input structure:
    - The board is 15x15 or 20x20 with one dominant background color.
    - The largest non-background monochrome object is the outer frame; inside its bbox is a bg moat and
      a smaller child frame, and both frames contain tiny 1-3-cell holes.
    - Outside the outer bbox are separated micro-sprites whose shapes match the hole masks up to
      rotation/flip.

    Rule:
    - Crop to the bbox of the largest monochrome component to get the full nested template.
    - Within that crop, detect every monochrome frame component and collect only those bg components whose
      whole 4-neighbor boundary is the current frame color; these are the true holes for that frame.
    - For each such hole, find all outside sprites with the same shape under rotation/flip and fill the
      hole with the unique plurality color for that shape, leaving the moat and child-frame structure
      unchanged.

    Key insight:
    - Shape matching is still the color key, but the crop is no longer a single flat frame: the solver
      must preserve the nested frame hierarchy and fill only bg pockets owned by each frame.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    outer = max(comps, key=lambda comp: int(comp["area"]))
    r0, c0, r1, c1 = map(int, outer["bbox"])

    sub = crop_bbox(g, outer["bbox"])

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    sprite_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert sprite_comps

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in sprite_comps:
        coords = comp["coords"]
        (_, _), mask = _coords_to_mask(coords)
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        for key in _all_mask_keys(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    frame_comps = find_components(sub, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert frame_comps

    out = sub.copy()
    for frame in sorted(frame_comps, key=lambda comp: int(comp["area"]), reverse=True):
        fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
        frame_col = int(frame["color"])
        frame_crop = crop_bbox(sub, frame["bbox"])
        for key, (hr0, hc0), hmask in _extract_owned_holes(frame_crop, frame_col, bg):
            cmap = sprites_by_key.get(key)
            assert cmap is not None
            fill_col = _plurality_color(cmap)
            patch = np.full(hmask.shape, fill_col, dtype=int)
            paste(out, patch, fr0 + hr0, fc0 + hc0, mask=hmask)

    return out


def _fits_some_orientation(shape: np.ndarray, occ: np.ndarray) -> bool:
    th, tw = occ.shape
    for oriented in _all_orientations(shape):
        sh, sw = oriented.shape
        if sh <= th - 2 and sw <= tw - 2:
            return True
    return False


def _validate_single_position(occ: np.ndarray, shape: np.ndarray, r: int, c: int):
    th, tw = occ.shape
    sh, sw = shape.shape
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return False
    new_cells = set()
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                new_cells.add((r + dr, c + dc))
    for rr, cc in new_cells:
        if not occ[rr, cc]:
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                return False
    return True


def _carve_single(occ: np.ndarray, shape: np.ndarray, r: int, c: int):
    sh, sw = shape.shape
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                occ[r + dr, c + dc] = False


def _place_single_hole(occ: np.ndarray, canonical_shape: np.ndarray):
    th, tw = occ.shape
    orientations = _all_orientations(canonical_shape)
    for _ in range(PLACEMENT_RETRIES):
        shape = random.choice(orientations)
        sh, sw = shape.shape
        if sh > th - 2 or sw > tw - 2:
            continue
        r = random.randint(1, th - sh - 1)
        c = random.randint(1, tw - sw - 1)
        if _validate_single_position(occ, shape, r, c):
            _carve_single(occ, shape, r, c)
            return shape, (r, c)
    return None


def _place_level_holes(
    occ: np.ndarray,
    shapes: list[np.ndarray],
    mult_range: tuple[int, int],
    min_total_holes: int,
):
    placements: dict[tuple[tuple[int, int], bytes], list[tuple[np.ndarray, tuple[int, int]]]] = {}
    total = 0
    for shape in shapes:
        key = _canonical_key(shape)
        count = random.randint(mult_range[0], mult_range[1])
        placed: list[tuple[np.ndarray, tuple[int, int]]] = []
        for _ in range(count):
            result = _place_single_hole(occ, shape)
            if result is not None:
                placed.append(result)
        if not placed:
            return None
        placements[key] = placed
        total += len(placed)
    if total < min_total_holes:
        return None
    return placements


def _make_outer_occ(th: int, tw: int, inner_r0: int, inner_c0: int, ih: int, iw: int):
    occ = np.ones((th, tw), dtype=bool)
    moat_r0 = inner_r0 - MOAT_GAP
    moat_c0 = inner_c0 - MOAT_GAP
    moat_r1 = inner_r0 + ih + MOAT_GAP - 1
    moat_c1 = inner_c0 + iw + MOAT_GAP - 1
    occ[moat_r0 : moat_r1 + 1, moat_c0 : moat_c1 + 1] = False
    return occ


def _render_base_patch(
    th: int,
    tw: int,
    bg: int,
    outer_col: int,
    inner_col: int,
    outer_occ: np.ndarray,
    inner_occ: np.ndarray,
    inner_origin: tuple[int, int],
):
    patch = np.full((th, tw), bg, dtype=int)
    patch[outer_occ] = outer_col
    ir0, ic0 = inner_origin
    inner_patch = np.full(inner_occ.shape, bg, dtype=int)
    inner_patch[inner_occ] = inner_col
    paste(patch, inner_patch, ir0, ic0)
    return patch


def _render_patch(
    th: int,
    tw: int,
    bg: int,
    outer_col: int,
    inner_col: int,
    outer_occ: np.ndarray,
    inner_occ: np.ndarray,
    inner_origin: tuple[int, int],
    color_by_key: dict[tuple[tuple[int, int], bytes], int],
    outer_holes,
    inner_holes,
):
    patch = _render_base_patch(th, tw, bg, outer_col, inner_col, outer_occ, inner_occ, inner_origin)

    for holes, origin in ((outer_holes, (0, 0)), (inner_holes, inner_origin)):
        or0, oc0 = origin
        for key, placed_list in holes.items():
            fill_col = int(color_by_key[key])
            for oriented, (r, c) in placed_list:
                fill_patch = np.full(oriented.shape, fill_col, dtype=int)
                paste(patch, fill_patch, or0 + r, oc0 + c, mask=oriented)
    return patch


def _choose_frame_origin(grid_size: int, th: int, tw: int):
    r_mid = (grid_size - th) // 2
    c_mid = (grid_size - tw) // 2
    r_lo = max(0, r_mid - 1)
    r_hi = min(grid_size - th, r_mid + 1)
    c_lo = max(0, c_mid - 1)
    c_hi = min(grid_size - tw, c_mid + 1)
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _place_sprite(inp: np.ndarray, placement: np.ndarray, shape: np.ndarray, color: int, bg: int):
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(
    inp: np.ndarray,
    shape_keys: list[tuple[tuple[int, int], bytes]],
    shape_lookup: dict[tuple[tuple[int, int], bytes], np.ndarray],
    color_by_key: dict[tuple[tuple[int, int], bytes], int],
    bg: int,
    forbidden_colors: tuple[int, ...],
    blocked_bbox: tuple[int, int, int, int],
):
    placement = inp.copy()
    br0, bc0, br1, bc1 = blocked_bbox
    placement[br0 : br1 + 1, bc0 : bc1 + 1] = (bg + 1) % NUM_COLORS
    other_pool = [color for color in range(NUM_COLORS) if color not in forbidden_colors and color not in color_by_key.values()]

    for key in shape_keys:
        base_shape = shape_lookup[key]
        orientations = _all_orientations(base_shape)
        fill_color = int(color_by_key[key])
        n_win = random.randint(*WINNING_SPRITE_RANGE)
        loser_color = random.choice(other_pool) if other_pool and random.random() < 0.8 else None

        for _ in range(n_win):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        if loser_color is not None:
            for _ in range(LOSER_SPRITE_COUNT):
                if not _place_sprite(inp, placement, random.choice(orientations), int(loser_color), bg):
                    return False
    return True


def _build_nested_layout(grid_size: int):
    cfg = INNER_SIZE_RANGES[grid_size]
    ih = random.randint(*cfg["h"])
    iw = random.randint(*cfg["w"])
    shell_r = random.randint(*cfg["shell"])
    shell_c = random.randint(*cfg["shell"])

    th = ih + 2 * shell_r + 2 * MOAT_GAP
    tw = iw + 2 * shell_c + 2 * MOAT_GAP
    inner_r0 = shell_r + MOAT_GAP
    inner_c0 = shell_c + MOAT_GAP

    outer_occ = _make_outer_occ(th, tw, inner_r0, inner_c0, ih, iw)
    inner_occ = np.ones((ih, iw), dtype=bool)
    return (th, tw), (inner_r0, inner_c0), outer_occ, inner_occ


def _build_shape_assignment(grid_size: int, outer_occ: np.ndarray, inner_occ: np.ndarray):
    cfg = LEVEL_CONFIG[grid_size]
    outer_shape_range = cfg["outer_shape_count"]
    inner_shape_range = cfg["inner_shape_count"]

    available = [shape for shape in SHAPE_LIBRARY if _fits_some_orientation(shape, outer_occ)]
    if len(available) < outer_shape_range[0] + inner_shape_range[0]:
        return None

    random.shuffle(available)
    outer_count = random.randint(*outer_shape_range)
    outer_shapes = available[:outer_count]
    remaining = [shape for shape in available if _canonical_key(shape) not in {_canonical_key(s) for s in outer_shapes}]
    inner_candidates = [shape for shape in remaining if _fits_some_orientation(shape, inner_occ)]
    if len(inner_candidates) < inner_shape_range[0]:
        return None
    inner_count = min(len(inner_candidates), random.randint(*inner_shape_range))
    inner_shapes = inner_candidates[:inner_count]

    outer_layout = _place_level_holes(
        outer_occ.copy(),
        outer_shapes,
        cfg["outer_mult"],
        min_total_holes=int(cfg["outer_min_holes"]),
    )
    if outer_layout is None:
        return None
    inner_layout = _place_level_holes(
        inner_occ.copy(),
        inner_shapes,
        cfg["inner_mult"],
        min_total_holes=int(cfg["inner_min_holes"]),
    )
    if inner_layout is None:
        return None
    return outer_layout, inner_layout, outer_shapes, inner_shapes


def _next_grid_size() -> int:
    global _GRID_SIZE_CURSOR
    grid_size = GRID_SIZE_OPTIONS[_GRID_SIZE_CURSOR % len(GRID_SIZE_OPTIONS)]
    _GRID_SIZE_CURSOR += 1
    return int(grid_size)


def _gen_attempt(grid_size: int) -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    outer_col, inner_col = random.sample([c for c in range(NUM_COLORS) if c != bg], 2)

    layout = _build_nested_layout(grid_size)
    (th, tw), inner_origin, outer_occ_base, inner_occ_base = layout

    shape_assignment = _build_shape_assignment(grid_size, outer_occ_base, inner_occ_base)
    if shape_assignment is None:
        return None
    outer_holes, inner_holes, outer_shapes, inner_shapes = shape_assignment

    outer_occ = outer_occ_base.copy()
    for placed_list in outer_holes.values():
        for oriented, (r, c) in placed_list:
            _carve_single(outer_occ, oriented, r, c)

    inner_occ = inner_occ_base.copy()
    for placed_list in inner_holes.values():
        for oriented, (r, c) in placed_list:
            _carve_single(inner_occ, oriented, r, c)

    shape_keys = list(outer_holes.keys()) + list(inner_holes.keys())
    avail_colors = [c for c in range(NUM_COLORS) if c not in (bg, outer_col, inner_col)]
    if len(avail_colors) < len(shape_keys):
        return None
    random.shuffle(avail_colors)
    color_by_key = {key: avail_colors[idx] for idx, key in enumerate(shape_keys)}

    output = _render_patch(
        th,
        tw,
        bg,
        outer_col,
        inner_col,
        outer_occ,
        inner_occ,
        inner_origin,
        color_by_key,
        outer_holes,
        inner_holes,
    )

    input_patch = _render_base_patch(th, tw, bg, outer_col, inner_col, outer_occ, inner_occ, inner_origin)

    fr, fc = _choose_frame_origin(grid_size, th, tw)
    inp = np.full((grid_size, grid_size), bg, dtype=int)
    paste(inp, input_patch, fr, fc)

    shape_lookup = {_canonical_key(shape): shape for shape in outer_shapes + inner_shapes}
    if not _scatter_sprites(
        inp,
        shape_keys,
        shape_lookup,
        color_by_key,
        bg,
        (bg, outer_col, inner_col),
        (fr, fc, fr + th - 1, fc + tw - 1),
    ):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    grid_size = _next_grid_size()
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt(grid_size)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
