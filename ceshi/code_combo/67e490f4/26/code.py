# Combo variant of 67e490f4 (index 26)
# combo_spec: D2/1+D4/1
# Name: 颜色编码旋转+仅旋转

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
# -------------------------------------
# spec_variant:
#   fusion_rule:
#     - Keep the original frame-with-holes backbone.
#     - Outside sprites are still monochrome connected components, but now their color has a dual role:
#       it is both the output fill color and an instruction for how many quarter-turns to rotate the
#       sprite before testing whether it matches a hole.
#     - Matching is rotation-only. Reflections never count.
#   traced_example:
#     - Fix the angle palette:
#         red(2) -> 0 turns, blue(1) -> 1 turn, green(3) -> 2 turns, yellow(4) -> 3 turns.
#     - Let the canonical outside sprite be the left-handed L:
#           10
#           11
#       A blue copy means "rotate this sprite by 90 degrees before matching". It therefore matches:
#           11
#           10
#       A green copy matches the 180-degree L, and so on.
#     - Under the old task, any color of that L would match any rotated L hole, because color only
#       supplied the fill color and shape matching itself could try all rotations. In the fused task,
#       only the color whose coded rotation lands exactly on the hole is valid.
#     - Add the mirrored J:
#           01
#           11
#       Because flips are forbidden, an L family and a J family stay distinct even if a reflection
#       would align them. This makes the D2 and D4 changes interact rather than operate independently.
#   complexity_calibration:
#     - The generator always includes a chiral mirror pair, so "no flip" is operational instead of
#       being a silent implementation detail.
#     - For each chosen canonical shape, the frame contains several different orientations of that
#       same shape. This gives repeated evidence that output color tracks orientation class.
#     - Outside sprites are provided in all four angle-colors with equal multiplicity per canonical
#       shape. A solver that ignores color-coded rotation collapses these families and fails on most
#       orientations, so the fusion cannot degenerate back to the original behavior.
#   implicit_assumption_review:
#     - STEP 1 still assumes the frame is the unique largest monochrome non-bg component; the
#       generator keeps the frame much larger than any outside sprite.
#     - STEP 2 keys each hole by its exact oriented mask. This is necessary now because two holes
#       with the same rotation-class but different orientations should map to different colors.
#     - The original STEP 3 assumption "try all rotations/flips for every sprite" is invalid.
#       Instead, each sprite contributes exactly one interpreted orientation: rotate the raw mask by
#       the number of quarter-turns encoded by the sprite's own color.
#     - The new solver assumes every sprite color is in the fixed angle palette and every hole has at
#       least one supporting sprite. The generator enforces both.
#   placement_legality:
#     - Holes stay strictly inside the frame bbox, never touch one another in 4-neighborhood, and
#       never touch the bbox border.
#     - Sprites are scattered with Chebyshev padding 1, so connected components do not merge outside.
#   visual_observability:
#     - Humans can see the same canonical outside shape repeated in red/blue/green/yellow while the
#       corresponding holes inside the frame appear in multiple orientations. The stable pattern is:
#       the fill color tells how far that outside shape had to be turned.
#     - The frame also contains both members of a mirror pair, so a reflection-sensitive shortcut is
#       visibly unsupported by the examples.
#   cleanliness:
#     - The scene still contains just one large frame plus separated small sprites.
#     - Only four sprite colors are used for the coded channel, which keeps the color-angle relation
#       readable from 2-3 examples instead of turning into unconstrained recoloring.


ANGLE_TO_COLOR = {0: 2, 1: 1, 2: 3, 3: 4}  # red, blue, green, yellow
COLOR_TO_TURNS = {color: turns for turns, color in ANGLE_TO_COLOR.items()}
ANGLE_COLORS = tuple(ANGLE_TO_COLOR[k] for k in range(4))


def _all_rotations(mask: np.ndarray):
    """Yield (key, oriented_mask) for the 4 rotational variants of a bool mask."""
    m0 = mask.astype(bool)
    for turns in range(4):
        m = np.rot90(m0, turns)
        yield (m.shape, m.tobytes()), m


def _rotation_key(mask: np.ndarray):
    """Canonical representative under rotations only."""
    return min(key for key, _ in _all_rotations(mask))


def _mirror_key(mask: np.ndarray):
    """Canonical key of the horizontally mirrored partner under rotation-only equivalence."""
    return _rotation_key(np.fliplr(mask.astype(bool)))


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Build a cropped bool mask from component coordinates."""
    r0, c0, r1, c1 = map(int, bbox)
    hh = int(r1 - r0 + 1)
    ww = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0])
    m = np.zeros((hh, ww), dtype=bool)
    m[rel[:, 0], rel[:, 1]] = True
    return m


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside a frame by decoding both sprite shape and sprite color-coded rotation.

    Input structure:
    - The most common color is the background. One large monochrome frame sits on it.
    - Inside the frame bbox are several hole components, seen as non-frame-color cells.
    - Outside the frame are small monochrome sprites. Their shape is a template, and their color is
      restricted to a fixed four-color angle code.

    Rule:
    - Crop to the frame bbox and start from a solid rectangle of the frame color.
    - For each outside sprite, rotate its mask by the number of quarter-turns specified by its color
      (red=0, blue=90, green=180, yellow=270), with no reflection allowed anywhere in matching.
    - For each hole, count only sprites whose color-decoded rotated mask matches that exact hole
      orientation, then fill the hole with the most frequent matching sprite color.

    Key insight:
    - A sprite's color is not just the paint to copy. It also tells you how far that sprite must be
      turned before comparison, so geometry and color have to be decoded together, and mirror images
      remain distinct because flips are never permitted.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the frame as the unique largest monochrome non-bg component.
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

    # STEP 2) Extract frame holes by their exact in-frame orientation.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hb = bbox_coords(coords)
        hr0, hc0, _, _ = map(int, hb)
        m = _mask_from_coords(coords, hb)
        holes.append((_mask_key(m), (int(hr0), int(hc0)), m))

    # STEP 3) Extract outside sprites and decode their single allowed rotated interpretation.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        bb = comp["bbox"]
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        assert scol in COLOR_TO_TURNS

        raw_mask = _mask_from_coords(coords, bb)
        decoded = np.rot90(raw_mask, COLOR_TO_TURNS[scol])
        key = _mask_key(decoded)
        bucket = sprites_by_key.setdefault(key, {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render the cropped output by filling each hole from its decoded sprite family.
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

FRAME_H_RANGE = (11, 15)
FRAME_W_RANGE = (12, 18)
FRAME_DENSITY_RANGE = (0.72, 0.88)

PLACEMENT_RETRIES = 250
SPRITE_PADDING = 1
SPRITES_PER_ANGLE_RANGE = (2, 3)
MAX_GEN_ATTEMPTS = 500


def _build_shape_library():
    raw = [
        [[1, 0], [1, 1]],
        [[0, 1], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1], [0, 1], [1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 1], [0, 1], [0, 1]],
        [[1, 1], [1, 0], [1, 0]],
    ]
    seen, dedup = set(), []
    for s in raw:
        m = np.array(s, dtype=bool)
        keys = [_mask_key(rot) for _, rot in _all_rotations(m)]
        if len(set(keys)) < 4:
            continue
        k = _rotation_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()
SHAPE_BY_KEY = {_rotation_key(shape): shape for shape in SHAPE_LIBRARY}


def _build_chiral_pairs():
    pairs = []
    seen = set()
    for shape in SHAPE_LIBRARY:
        key = _rotation_key(shape)
        mirror = _mirror_key(shape)
        if mirror == key or mirror not in SHAPE_BY_KEY:
            continue
        pid = tuple(sorted((key, mirror)))
        if pid in seen:
            continue
        seen.add(pid)
        pairs.append((SHAPE_BY_KEY[pid[0]], SHAPE_BY_KEY[pid[1]]))
    return pairs


CHIRAL_PAIRS = _build_chiral_pairs()


def _plan_hole_families():
    assert CHIRAL_PAIRS
    left, right = random.choice(CHIRAL_PAIRS)
    pair_keys = {_rotation_key(left), _rotation_key(right)}
    extras = [s for s in SHAPE_LIBRARY if _rotation_key(s) not in pair_keys]
    extra = random.choice(extras) if extras else None

    base_shapes = [left, right]
    if extra is not None:
        base_shapes.append(extra)

    hole_specs = []
    for shape in (left, right):
        for turns in range(4):
            hole_specs.append({"base_shape": shape, "turns": turns, "mask": np.rot90(shape, turns)})

    if extra is not None:
        for turns in random.sample(range(4), 2):
            hole_specs.append({"base_shape": extra, "turns": turns, "mask": np.rot90(extra, turns)})

    random.shuffle(hole_specs)
    return base_shapes, hole_specs


def _can_place_hole(occ: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    hh, ww = mask.shape
    if top < 1 or left < 1 or top + hh > occ.shape[0] - 1 or left + ww > occ.shape[1] - 1:
        return False

    new_cells = {
        (top + dr, left + dc)
        for dr in range(hh)
        for dc in range(ww)
        if mask[dr, dc]
    }
    for rr, cc in new_cells:
        if not occ[rr, cc]:
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < occ.shape[0] and 0 <= nc < occ.shape[1] and not occ[nr, nc]:
                if (nr, nc) not in new_cells:
                    return False
    return True


def _carve_hole(occ: np.ndarray, mask: np.ndarray, top: int, left: int):
    hh, ww = mask.shape
    for dr in range(hh):
        for dc in range(ww):
            if mask[dr, dc]:
                occ[top + dr, left + dc] = False


def _place_holes(th: int, tw: int, hole_specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    occ = np.ones((th, tw), dtype=bool)
    order = sorted(
        hole_specs,
        key=lambda spec: (-int(np.sum(spec["mask"])), -(spec["mask"].shape[0] * spec["mask"].shape[1]), random.random()),
    )
    placements: list[dict] = []

    for spec in order:
        mask = spec["mask"]
        hh, ww = mask.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(1, th - hh - 1)
            left = random.randint(1, tw - ww - 1)
            if _can_place_hole(occ, mask, top, left):
                _carve_hole(occ, mask, top, left)
                placements.append({**spec, "pos": (top, left)})
                placed = True
                break
        if not placed:
            return None

    density = float(np.mean(occ))
    if not (FRAME_DENSITY_RANGE[0] <= density <= FRAME_DENSITY_RANGE[1]):
        return None

    return occ, placements


def _build_output(occ: np.ndarray, placements: list[dict], frame_col: int) -> np.ndarray:
    out = np.full(occ.shape, int(frame_col), dtype=int)
    for rec in placements:
        turns = int(rec["turns"])
        color = int(ANGLE_TO_COLOR[turns])
        top, left = rec["pos"]
        mask = rec["mask"]
        patch = np.full(mask.shape, color, dtype=int)
        paste(out, patch, int(top), int(left), mask=mask)
    return out


def _place_sprite(inp: np.ndarray, placement: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    try:
        top, left = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, top, left, transparent=bg)
    paste(inp, patch, top, left, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, base_shapes: list[np.ndarray], bg: int, frame_col: int) -> bool:
    del frame_col  # kept in the signature for symmetry with the original scaffold
    placement = inp.copy()
    copies_per_angle = random.randint(*SPRITES_PER_ANGLE_RANGE)

    for shape in base_shapes:
        for turns in range(4):
            color = int(ANGLE_TO_COLOR[turns])
            for _ in range(copies_per_angle):
                if not _place_sprite(inp, placement, shape, color, bg):
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg_candidates = [c for c in range(NUM_COLORS) if c not in ANGLE_COLORS]
    bg = random.choice(bg_candidates)
    frame_col = random.choice([c for c in bg_candidates if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    base_shapes, hole_specs = _plan_hole_families()
    layout = _place_holes(th, tw, hole_specs)
    if layout is None:
        return None
    occ, placements = layout

    output = _build_output(occ, placements, frame_col)

    top = random.randint(0, GRID_SIZE - th)
    left = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), top, left)

    if not _scatter_sprites(inp, base_shapes, bg, frame_col):
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
