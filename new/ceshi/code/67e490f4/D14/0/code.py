# Variant of 67e490f4
# Dimension: D14 — 框架色的角色 (meta)
# Change: 框架色为单一非背景色，在输出中填充所有非洞区域，构成实心模板
#         → 框架色为两色交替的模板图案，洞定义为“不属于这两种框架色”的区域，输出保留该双色图案

"""
Design analysis for the D14 variant
==================================

Source-task check:
- source_task is None in the prompt, so there is no extra reference solver to read.

a) Concretization and behavior trace
- Original visual structure: one large monochrome frame with bg-colored holes, plus outside sprites.
- Variant visual structure: the large frame is no longer monochrome. Its non-hole cells are filled by
  exactly two alternating colors, using a simple visible motif such as checkerboard, row stripes, or
  column stripes. Holes are still bg in the input, and outside sprites still encode hole-fill colors.
- Small trace:
  old input crop:
    2 2 2 2
    2 0 0 2
    2 2 2 2
  old solver used frame_col=2, so hole_mask = (sub != 2).
  new input crop:
    2 3 2 3
    3 0 0 2
    2 3 2 3
  now the correct hole_mask is cells that are neither 2 nor 3. The old `sub != frame_col` logic would
  incorrectly treat every color-3 frame cell as a hole. The new solver must first infer the pair
  {2,3}, then define holes by exclusion from that pair, and finally start the output from the input's
  cropped two-color template instead of a solid fill.

b) Complexity calibration
- The core novelty must be visible in 2-3 examples, so the frame always uses exactly two colors and
  they alternate across the whole bbox. This is richer than the original single-color frame, but still
  low-complexity: the solver only needs to identify a color pair, not an arbitrary texture palette.
- Pattern style is chosen from a tiny family (`checker`, `row`, `col`) so examples are not repetitive
  while still being easy to parse visually.
- Sprite colors explicitly exclude both frame colors. This avoids ambiguous "is this a frame color or a
  sprite color?" situations and keeps the D14 change focused on the frame itself.

c) Implicit-assumption audit against the original `main()`
- `find_components(... monochromatic=True)` for the frame relied on "the frame is one color". This no
  longer holds, so frame detection must search over pairs of colors and use their union mask.
- `frame["color"]` assumed a single frame color id. The variant has two frame colors, so the solver
  stores `frame_cols`.
- `hole_mask = (sub != frame_col)` depended on the second frame color not existing. In the variant this
  would destroy correctness immediately; it must become `~np.isin(sub, frame_cols)`.
- `out = np.full((th, tw), frame_col, ...)` would overwrite the intended two-color pattern. The output
  must instead begin as `sub.copy()` and only replace hole cells.
- The original tie-break "smaller color id wins" is a hidden, non-visual rule. The generator therefore
  guarantees a unique most frequent sprite color for each shape, and the solver asserts uniqueness.

d) Placement legality
- The frame bbox still stays inside the 30x30 canvas.
- Hole placement logic from the original task is preserved: holes remain strictly inside the bbox border,
  4-disconnected from each other, and symmetric. This guarantees the frame remains one large connected
  two-color template under 8-connectivity.
- Outside sprites are still separated by at least one bg cell and are placed only after the entire frame
  bbox is pasted, so they never overlap the template.

e) Visual observability
- The D14 change is obvious because the output is no longer a solid-colored slab: the template's
  alternating two-color structure remains visible everywhere except inside the filled holes.
- The rule is consistent across examples: "treat both alternating frame colors as the template material,
  treat any other cell inside the bbox as a hole."

f) Visual cleanliness
- Only one large structured object occupies the main area; sprites stay outside it with padding.
- The frame uses exactly two colors, hole fills use different colors, and holes never touch the border,
  so the frame boundary remains easy to read.

g) Color-change inferability
- No arithmetic or id-based color transformation is used.
- Hole-fill colors are inferred exactly as before from sprite plurality per shape, while frame colors are
  copied from the visible template pattern itself.

Generator/solver contract after modification
- The solver now assumes the two frame colors are not reused by sprites. The generator enforces this.
- The solver assumes the true frame pair forms the largest valid two-color connected component whose bbox
  border is fully covered by those two colors and whose bbox contains at least one non-pair cell. The
  generator enforces this by constructing one large patterned frame and keeping all sprites off-frame.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
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


def _all_mask_keys(mask: np.ndarray):
    """Generate keys for all rotations and vertical-flip variants of a boolean mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes())


def _border_is_full(mask: np.ndarray) -> bool:
    """Return True iff every bbox-border cell is part of the patterned frame."""
    return bool(
        mask[0, :].all()
        and mask[-1, :].all()
        and mask[:, 0].all()
        and mask[:, -1].all()
    )


def _find_patterned_frame(g: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Locate the large two-color frame as the strongest valid color-pair component."""
    colors = sorted(set(map(int, g.flatten())) - {int(bg)})
    assert len(colors) >= 2

    best_score = None
    best_pair = None
    best_bbox = None

    for cols in combinations(colors, 2):
        pair_mask = np.isin(g, cols)
        if not bool(pair_mask.any()):
            continue
        for coords in components_mask(pair_mask, connectivity=8):
            area = int(coords.shape[0])
            bbox = bbox_coords(coords)
            sub = crop_bbox(g, bbox)
            sub_pair_mask = np.isin(sub, cols)
            if not _border_is_full(sub_pair_mask):
                continue
            hole_area = int((~sub_pair_mask).sum())
            if hole_area <= 0:
                continue
            bbox_area = int(sub.shape[0] * sub.shape[1])
            score = (area, bbox_area, hole_area)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (int(cols[0]), int(cols[1]))
                best_bbox = bbox

    assert best_pair is not None and best_bbox is not None
    return best_pair, best_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a holed template whose material alternates between two frame colors.

    Input structure:
    - bg is the most common color. One large frame/template is made from exactly two alternating colors.
    - Inside that frame bbox, true holes are the connected regions that use neither frame color.
    - Outside the frame bbox, scattered monochrome sprites provide shape-coded color evidence.

    Rule:
    - Find the large two-color frame and crop to its bounding box, preserving its visible pattern.
    - Detect each hole as a 4-connected shape formed by cells that are not either frame color.
    - Match each hole shape against outside sprites up to rotation/flip, and fill the hole with the
      unique most frequent sprite color for that shape.

    Key insight: The template material is now a two-color pattern rather than one solid color, so the
    solver must first infer the frame color pair and keep that pattern unchanged while only recoloring the holes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the patterned frame as the largest valid component formed by a pair of colors.
    frame_cols, fb = _find_patterned_frame(g, bg)
    r0, c0, r1, c1 = map(int, fb)
    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    # STEP 2) Extract hole components inside the bbox (anything that is neither frame color), 4-connected.
    frame_mask = np.isin(sub, frame_cols)
    hole_mask = ~frame_mask
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        key = (m.shape, m.tobytes())
        holes.append((key, (int(hr0), int(hc0)), m))

    # STEP 3) Extract sprites outside the frame bbox.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        bb = comp["bbox"]
        sr0, sc0, sr1, sc1 = map(int, bb)
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        if scol in frame_cols:
            continue

        rel = coords - np.array([sr0, sc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(m):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render output: preserve the two-color template, then fill each hole by shape match.
    out = sub.copy()
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        max_count = max(cmap.values())
        winners = [color for color, count in cmap.items() if count == max_count]
        assert len(winners) == 1
        scol = int(winners[0])
        patch = np.full(hmask.shape, scol, dtype=int)
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

# Frame appearance
PATTERN_STYLES = ("checker", "row", "col")

# Sprite scattering
WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 400


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants (canonical representative)."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
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
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4-fold symmetric mirror positions (deduplicated)."""
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap with existing holes, and 4-disconnected from them."""
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
    """Place shape in the top-left quadrant, then mirror to all four quadrants."""
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    """Place on the horizontal center line, mirrored left-right."""
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
    c_m = tw - sw - c
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ, shape, th, tw):
    """Place on the vertical center line, mirrored up-down."""
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
    r_m = th - sh - r
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ, shape, th, tw):
    """Place at dead center. Requires both dims odd."""
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _try_place_symmetric(occ, shape, th, tw):
    """Try one of the symmetric placement modes."""
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    """Mark hole cells as unavailable in the occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place holes with LR+UD+rot180 symmetry. Returns (occ, hole_placements) or None."""
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


def _make_frame_pattern(th: int, tw: int, frame_cols: tuple[int, int]):
    """Create the visible two-color frame pattern for the template bbox."""
    a, b = map(int, frame_cols)
    rr, cc = np.indices((th, tw))
    style = random.choice(PATTERN_STYLES)
    if style == "checker":
        selector = (rr + cc) % 2 == 0
    elif style == "row":
        selector = rr % 2 == 0
    else:
        selector = cc % 2 == 0
    pattern = np.where(selector, a, b).astype(int)
    return pattern, style


def _build_output(occ, hole_placements, frame_cols, bg):
    """Render the target output and return (output, shape_info, pattern_style)."""
    th, tw = occ.shape
    disallowed = {int(bg), int(frame_cols[0]), int(frame_cols[1])}
    avail_colors = [c for c in range(NUM_COLORS) if c not in disallowed]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output, style = _make_frame_pattern(th, tw, frame_cols)
    shape_info = []
    for (shape, pls), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in pls:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info, style


def _place_sprite(inp, placement, ori, color, bg):
    """Place a single sprite on the canvas. Returns True on success."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_cols):
    """Place winning and losing sprites for each hole shape outside the frame."""
    placement = inp.copy()
    disallowed = {int(bg), int(frame_cols[0]), int(frame_cols[1])}
    other_pool = [c for c in range(NUM_COLORS) if c not in disallowed]

    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, min(n_other, len(losers)))

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        for oc in oc_list:
            n_oc = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build patterned frame, scatter sprites, validate."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_cols = tuple(random.sample([c for c in range(NUM_COLORS) if c != bg], 2))
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    result = _build_output(occ, hole_placements, frame_cols, bg)
    if result is None:
        return None
    output, shape_info, _style = result

    frame_pattern = output.copy()
    shape_colors = {fill_color for _, fill_color, _ in shape_info}
    frame_only = frame_pattern.copy()
    frame_only[np.isin(frame_only, list(shape_colors))] = bg

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_only, bg).astype(int), fr, fc)

    if not _scatter_sprites(inp, shape_info, bg, frame_cols):
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
