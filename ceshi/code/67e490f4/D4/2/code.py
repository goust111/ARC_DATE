# Variant of 67e490f4
# Dimension: D4 — 源对象：精灵的颜色特征 (src)
# Change: 精灵为单色连通分量（每个精灵所有像素同色），6-7种不同精灵颜色，均不等于背景色和框架色... → 精灵含背景色内洞

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

# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior:
# - The frame logic stays the same: one large monochrome frame contains many bg holes.
# - The output still fills each frame-hole with a color chosen from matching outside sprites.
# - The change is on the source sprites: each outside sprite now has a solid outer silhouette
#   but one or more enclosed bg cells carved out of its interior. Example:
#     original source:  333      variant source:  333
#                       333                       303
#                       333                       333
#   The frame-hole is the solid 3x3 square, but the outside sprite is a hollow 3x3 ring.
#   Therefore exact foreground masks no longer match; the solver must first fill enclosed bg
#   holes inside the sprite to recover its outer silhouette, then compare that silhouette with
#   the frame-hole shape under rotation/flip.
# - This affects the pipeline exactly at sprite parsing / shape-key extraction. Frame detection,
#   hole extraction, plurality-of-colors, and cropped output remain unchanged.
#
# b) Complexity calibration:
# - If sprites were allowed to stay solid, this variant would visually collapse back to the
#   original task. To prevent degeneration, every placed sprite is forced to contain at least
#   one enclosed bg cavity.
# - To make the abstraction observable, the hole-shape library uses thicker silhouettes that
#   actually admit interior cavities; very thin polyominoes would not show the change.
# - Each canonical hole shape appears multiple times and is paired with several sprite colors,
#   so the plurality rule still has repeated evidence instead of a single fragile clue.
#
# c) Hidden assumptions in the original solver that no longer hold:
# - Original STEP 3 used the raw sprite foreground mask as the matching key. That assumed a
#   sprite was a solid block and that "visible painted cells == semantic shape". This is false
#   now: the semantic key is the outer silhouette after filling internal cavities.
# - Original generation used `patch[ori] = color`, which also assumes no bg cells may exist
#   inside a sprite bbox. In the variant this must be replaced with hollow-shell rendering.
# - Color extraction via mode still works because every sprite remains monochrome; only bg
#   cavities are inserted, never extra painted colors.
#
# d) Placement legality:
# - Hole placement inside the frame still enforces border preservation, no overlap, and
#   4-disconnection between holes, so hole extraction in `main()` is unambiguous.
# - Sprite placement still uses `random_free_location(..., padding=1)`, so hollow sprites do
#   not touch each other or the frame. Internal cavities are fully enclosed and do not merge
#   with outside bg because the cavity cells never touch the sprite bbox boundary.
#
# e) Visual observability:
# - In 2-3 examples a human can now see that outside objects are not the same as the frame
#   holes cell-for-cell: the outside objects visibly contain bg holes, while the frame holes are
#   solid silhouettes. The consistent rule is "ignore the hollow interior; use the outer shell".
# - The plurality color rule remains visually inferable because multiple sprites share the same
#   outer silhouette but vary in color.
#
# f) Visual cleanliness:
# - Only two structure types coexist: one large frame and separated small sprites on empty bg.
#   Hollow details stay inside each sprite's own bbox, so they do not overlap other evidence.
# - The silhouette library avoids extremely large or hairline-thin shapes, keeping the grids
#   readable after several repeated placements.
#
# g) Color inference:
# - Colors are not transformed numerically. For each silhouette, the most frequent outside sprite
#   color directly becomes the fill color of matching frame holes. The extra abstraction comes
#   only from recovering the silhouette, not from any hidden color arithmetic.
#
# New solver assumptions to audit against the generator:
# - Every outside sprite must have at least one enclosed bg cavity; the generator enforces this.
# - Every frame-hole silhouette must have at least one matching outside sprite after cavity fill;
#   the generator creates winning sprites for every chosen silhouette.
# - A sprite's enclosed cavity must not touch its bbox border; cavity generation validates this.


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
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, out = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _fill_enclosed_bg(mask: np.ndarray) -> np.ndarray:
    """Fill bg pockets that are fully enclosed inside a silhouette bbox."""
    filled = mask.astype(bool).copy()
    bg_mask = ~filled
    if not bg_mask.any():
        return filled
    hh, ww = filled.shape
    for coords in components_mask(bg_mask, connectivity=4):
        touches_edge = False
        for rr, cc in coords:
            if int(rr) == 0 or int(cc) == 0 or int(rr) == hh - 1 or int(cc) == ww - 1:
                touches_edge = True
                break
        if not touches_edge:
            filled[coords[:, 0], coords[:, 1]] = True
    return filled


def _extract_mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return a cropped bool mask for coords within bbox."""
    r0, c0, r1, c1 = map(int, bbox)
    hh = int(r1 - r0 + 1)
    ww = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0])
    m = np.zeros((hh, ww), dtype=bool)
    m[rel[:, 0], rel[:, 1]] = True
    return m


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside a large frame using the outer silhouettes of hollow outside sprites.

    Input structure:
    - bg is the most common color. One large monochrome frame sits somewhere in the grid.
    - Inside the frame bbox are several bg holes whose solid silhouettes matter.
    - Outside the frame are small monochrome sprites; each sprite contains enclosed bg cavities.

    Rule:
    - Crop to the frame bbox and begin from a solid rectangle of the frame color.
    - For each frame-hole, inspect outside sprites after filling their enclosed bg cavities so
      only the outer silhouette remains; compare silhouettes up to rotation/flip.
    - Among sprites whose recovered silhouette matches the hole, use the most frequent sprite
      color to fill that hole in the output.

    Key insight:
    - The painted cells of a sprite are no longer its matching key. The key is the silhouette you
      get after mentally ignoring the hollow interior, while the plurality color of those sprites
      still determines the fill color for the corresponding frame-hole.
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

    # STEP 2) Extract frame-holes as raw solid silhouette masks.
    hole_mask = (sub != int(frame_col))
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hb = bbox_coords(coords)
        hr0, hc0, _, _ = map(int, hb)
        m = _extract_mask_from_coords(coords, hb)
        key = (m.shape, m.tobytes())
        holes.append((key, (int(hr0), int(hc0)), m))

    # STEP 3) Extract outside sprites and recover their outer silhouettes.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        bb = comp["bbox"]
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        raw_mask = _extract_mask_from_coords(coords, bb)
        silhouette = _fill_enclosed_bg(raw_mask)

        for key, _ in _all_rot_flip(silhouette):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render output from a solid frame-colored rectangle, then fill matching holes.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions; enlarged slightly because silhouettes are thicker than the original task.
FRAME_H_RANGE = (11, 17)
FRAME_W_RANGE = (11, 24)

# Hole layout
SHAPE_COUNT_RANGE = (3, 5)
MIN_SHAPES = 3
FRAME_DENSITY_RANGE = (0.55, 0.88)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering
WINNING_SPRITE_RANGE = (2, 4)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 400


def _build_shape_library():
    """Thick silhouettes that admit at least one enclosed interior cavity."""
    raw = [
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]],
    ]
    seen, out = set(), []
    for s in raw:
        m = np.array(s, dtype=bool)
        if not _cavity_options(m):
            continue
        key = _canonical_key(m)
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4-fold symmetric mirror positions (deduplicated)."""
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap, and 4-disconnected from existing holes."""
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


def _cavity_options(mask: np.ndarray) -> list[np.ndarray]:
    """All small enclosed cavity masks that preserve a single connected sprite."""
    mask = mask.astype(bool)
    hh, ww = mask.shape
    candidates = []
    for r in range(1, hh - 1):
        for c in range(1, ww - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                candidates.append((r, c))

    options: list[np.ndarray] = []
    seen = set()

    def try_add(cells):
        cavity = np.zeros_like(mask, dtype=bool)
        for r, c in cells:
            cavity[r, c] = True
        sprite = mask & ~cavity
        fg_comps = components_mask(sprite, connectivity=8)
        if len(fg_comps) != 1:
            return
        bg_comps = components_mask(~sprite, connectivity=4)
        enclosed = 0
        for coords in bg_comps:
            touches_edge = False
            for rr, cc in coords:
                if int(rr) == 0 or int(cc) == 0 or int(rr) == hh - 1 or int(cc) == ww - 1:
                    touches_edge = True
                    break
            if not touches_edge:
                enclosed += 1
        if enclosed < 1:
            return
        key = cavity.tobytes()
        if key not in seen:
            seen.add(key)
            options.append(cavity)

    for cell in candidates:
        try_add([cell])

    cand_set = set(candidates)
    for r, c in candidates:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) in cand_set:
                try_add([(r, c), (nr, nc)])

    return options


SHAPE_LIBRARY = _build_shape_library()


def _build_hole_layout(th, tw):
    """Place symmetric solid silhouettes that will later be keyed by hollow sprites."""
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [s for s in SHAPE_LIBRARY if s.shape[0] <= max_sh and s.shape[1] <= max_sw]
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


def _build_output(occ, hole_placements, frame_col, bg):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, pls), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in pls:
            for pr, pc in positions:
                patch = np.full(shape.shape, fill_col, dtype=int)
                paste(output, patch, pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info


def _render_hollow_sprite(shape: np.ndarray, color: int, bg: int) -> np.ndarray | None:
    """Render a single-color sprite with at least one enclosed bg cavity."""
    options = _cavity_options(shape)
    if not options:
        return None
    cavity = random.choice(options)
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = int(color)
    patch[cavity] = int(bg)
    return patch


def _place_sprite(inp, placement, shape, color, bg):
    patch = _render_hollow_sprite(shape, color, bg)
    if patch is None:
        return False
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_col):
    """Place hollow winning and losing sprites for each silhouette outside the frame."""
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

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

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

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
