# Combo variant of 67e490f4 (index 20)
# combo_spec: D7/2+D8/2
# Name: 旋转对称洞+旋转对称布局

from __future__ import annotations

# Design analysis for combo D7/2 + D8/2
# a) Concrete behavior and trace:
#    - Original 67e490f4 used a large frame with holes plus outside sprites. The semantic rule was already
#      "match a hole to sprites by shape, then fill that hole with the plurality sprite color".
#    - Two structural regularities change together here:
#      1. D7/2: every hole shape itself has non-trivial rotational symmetry, so there are no chiral L/T/Z
#         cavities anymore. Examples now include bars, squares, pluses, and 3x3 rings.
#      2. D8/2: the hole layout is no longer mirror-built. Instead the whole frame interior is a square whose
#         hole pattern is invariant under 90-degree rotation but intentionally rejects LR/UD mirror symmetry.
#    - Small trace: suppose a 1x5 bar hole is placed near the upper-left arc of the square. Rotating the whole
#      layout by 90 degrees creates three partner holes around the center, with horizontal bars becoming
#      vertical bars where needed. Outside the frame, several sprites of that same canonical family appear in
#      mixed colors; the most frequent one still decides the fill color for all four bar holes.
#    - Difference from the source task: the answer is still keyed by shape family, but the frame now looks like
#      a kaleidoscope. Humans can exploit that visible rot90 regularity while also noticing that every cavity
#      is itself a rotationally symmetric motif rather than an arbitrary irregular outline.
# b) Complexity calibration:
#    - Using only trivial 1x1 holes would collapse the variant, so each sample contains 3-4 shape families and
#      must include at least one non-rectangular 90-symmetric family (`plus` or `ring`).
#    - Each family appears on a 4-position rot90 orbit, giving repeated evidence without requiring hidden
#      ordering rules. Total holes are therefore usually 12 or 16.
#    - Sprite winner colors are strict pluralities, so the solver never needs the original numeric tie-break.
# c) Implicit-assumption audit of the original main():
#    - `find_components(... monochromatic=True)` assumes the frame is the largest monochrome non-bg component.
#      The generator keeps the square frame much larger than any sprite and enforces sprite/frame separation.
#    - `hole_mask = sub != frame_col` assumes the frame bbox contains only frame cells and holes. The generator
#      never puts clue sprites inside the frame bbox.
#    - Hole keys are exact 4-connected masks, while sprites are indexed under all rotations/flips. That still
#      works after fusion, but only if every hole family gets at least one sprite family outside the frame and
#      all orbit members really are rotational variants of the same family. The generator records placements by
#      canonical key and validates that.
#    - The original solver never checked symmetry. After fusion that becomes a generator responsibility:
#      generated hole masks must satisfy rot90 symmetry and must fail LR/UD symmetry, otherwise D8/2 would
#      silently degenerate back toward the original mirrored layout.
# d) Placement legality:
#    - The frame bbox is always square and stays in-bounds on the 30x30 canvas.
#    - Hole copies are created by rotating one placed component around the square center; validation rejects
#      overlap, border contact, and 4-neighbor contact between distinct holes.
#    - Outside sprites are placed with Chebyshev padding 1 using `random_free_location`, so they remain
#      separated from each other and from the frame.
# e) Visual observability:
#    - Across 2-4 samples, a human can plainly see the frame is a square pinwheel rather than an LR/UD mirror.
#    - The cavity inventory itself is visibly restricted to rotationally symmetric motifs, which distinguishes
#      the combo from the source task's more generic irregular holes.
# f) Visual cleanliness:
#    - There are only two visual systems: one square frame and scattered clue sprites. No extra markers or
#      overlays are added.
#    - Padding and moderate frame sizes keep the sprite field readable instead of crowding the frame edge.
# g) Change inferability:
#    - Colors are still inferred structurally from repeated shape clues, not from color ids or hidden ranks.
#    - The new symmetry rules are geometric and directly visible: one can verify them by eye from the hole
#      pattern itself, without any invisible metadata.
# h) Quantitative anti-degeneration metrics used after generation:
#    - `rot90_layout = 1` and `lr_layout = ud_layout = 0` for the hole mask inside the square frame.
#    - `all_holes_rot_sym = 1`, meaning every individual hole equals either its 90-degree or 180-degree
#      rotation after trimming to its own bbox.
# i) New solver assumptions and generator support:
#    - `main()` now effectively assumes the frame bbox is square and that every hole family has matching sprite
#      evidence. `generate_input()` guarantees both and rejects any oracle mismatch.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a square pinwheel frame by filling rotationally symmetric holes from outside sprite clues.

    Input structure:
    - The most common color is the background, and one large monochrome square frame is the dominant object.
    - Inside that square are several 4-connected holes whose individual shapes have rotational symmetry and
      whose overall placement forms a 90-degree rotational pattern rather than the source task's mirror layout.
    - Outside the frame, small separated sprites encode hole families by shape and candidate fill colors by
      their own monochrome color.

    Rule:
    - Find the large square frame, crop to its bounding box, and start from a solid block of the frame color.
    - Extract each hole as an exact 4-connected mask. Extract every outside sprite and index it under all
      rotations and reflections of its mask.
    - For each hole, look up the matching sprite family and paint the hole with the most frequent sprite color
      for that family.

    Key insight:
    - The new rot90 regularity is visually helpful but not itself the fill key; the actual answer still comes
      from shape-family matching between holes and outside sprites, with plurality color voting per family.
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
    assert (r1 - r0) == (c1 - c0)

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    hole_mask = sub != frame_col
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append(((m.shape, m.tobytes()), (int(hr0), int(hc0)), m))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

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

    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_SIDE_RANGE = (9, 13)
SHAPE_COUNT_OPTIONS = (3, 4)
SHAPE_COUNT_WEIGHTS = (2, 3)
FRAME_DENSITY_RANGE = (0.58, 0.84)
PLACEMENT_RETRIES = 180

WINNING_SPRITE_RANGE = (3, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 500


def _all_rot_flip(mask: np.ndarray):
    m0 = np.array(mask, dtype=bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _shape_has_rot_sym(mask: np.ndarray) -> bool:
    m = np.array(mask, dtype=bool)
    return bool(np.array_equal(m, np.rot90(m, 2)) or (m.shape[0] == m.shape[1] and np.array_equal(m, np.rot90(m, 1))))


def _shape_has_rot90(mask: np.ndarray) -> bool:
    m = np.array(mask, dtype=bool)
    return bool(m.shape[0] == m.shape[1] and np.array_equal(m, np.rot90(m, 1)))


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1, 1, 1]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    rect_like: list[np.ndarray] = []
    special: list[np.ndarray] = []
    for arr in raw:
        mask = np.array(arr, dtype=bool)
        assert _shape_has_rot_sym(mask)
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        if _shape_has_rot90(mask) and int(np.sum(mask)) not in {1, 4}:
            special.append(mask)
        else:
            rect_like.append(mask)
    return rect_like, special


RECT_LIBRARY, SPECIAL_LIBRARY = _build_shape_library()
SHAPE_LIBRARY = RECT_LIBRARY + SPECIAL_LIBRARY
FOUR_SHAPE_MIN_SIDE = 12
FOUR_SHAPE_MAX_AREA = 4
FOUR_SHAPE_MAX_DIM = 3


def _rot90_coords(coords: np.ndarray, side: int) -> np.ndarray:
    side = int(side)
    out = np.zeros_like(coords, dtype=int)
    out[:, 0] = coords[:, 1]
    out[:, 1] = side - 1 - coords[:, 0]
    return out


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    hh = int(r1 - r0 + 1)
    ww = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((hh, ww), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return (int(r0), int(c0)), mask


def _shape_orbit(shape: np.ndarray, r0: int, c0: int, side: int) -> list[np.ndarray] | None:
    pts = np.argwhere(shape)
    base = pts + np.array([int(r0), int(c0)])
    orbit = [base]
    for _ in range(3):
        orbit.append(_rot90_coords(orbit[-1], side))

    cell_to_group: dict[tuple[int, int], int] = {}
    for gi, coords in enumerate(orbit):
        if coords.size == 0:
            return None
        for rr, cc in coords:
            r = int(rr)
            c = int(cc)
            if r <= 0 or c <= 0 or r >= side - 1 or c >= side - 1:
                return None
            key = (r, c)
            if key in cell_to_group:
                return None
            cell_to_group[key] = gi

    for (r, c), gi in cell_to_group.items():
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            gj = cell_to_group.get((nr, nc))
            if gj is not None and gj != gi:
                return None
    return [np.array(coords, dtype=int) for coords in orbit]


def _orbit_fits(occ: np.ndarray, orbit: list[np.ndarray]) -> bool:
    th, tw = occ.shape
    assert th == tw
    cell_to_group: dict[tuple[int, int], int] = {}
    for gi, coords in enumerate(orbit):
        for rr, cc in coords:
            r = int(rr)
            c = int(cc)
            if not occ[r, c]:
                return False
            cell_to_group[(r, c)] = gi

    for (r, c), gi in cell_to_group.items():
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < th and 0 <= nc < tw:
                gj = cell_to_group.get((nr, nc))
                if gj is None:
                    if not occ[nr, nc]:
                        return False
                elif gj != gi:
                    return False
    return True


def _carve_orbit(occ: np.ndarray, orbit: list[np.ndarray]) -> None:
    for coords in orbit:
        occ[coords[:, 0], coords[:, 1]] = False


def _hole_mask_is_valid(hole_mask: np.ndarray) -> bool:
    return bool(
        np.array_equal(hole_mask, np.rot90(hole_mask, 1))
        and not np.array_equal(hole_mask, np.fliplr(hole_mask))
        and not np.array_equal(hole_mask, np.flipud(hole_mask))
    )


def _choose_shapes(side: int, k_target: int) -> list[np.ndarray]:
    special = [s for s in SPECIAL_LIBRARY if max(s.shape) <= side - 2]
    regular = [s for s in RECT_LIBRARY if max(s.shape) <= side - 2]
    assert special
    shapes = [random.choice(special)]
    if k_target >= 4:
        compact_regular = [
            s
            for s in regular
            if int(np.sum(s)) <= FOUR_SHAPE_MAX_AREA and max(s.shape) <= FOUR_SHAPE_MAX_DIM
        ]
        if len(compact_regular) >= k_target - 1:
            regular = compact_regular
    pool = regular + [s for s in special if _canonical_key(s) != _canonical_key(shapes[0])]
    random.shuffle(pool)
    for shape in pool:
        if len(shapes) >= k_target:
            break
        if _canonical_key(shape) not in {_canonical_key(x) for x in shapes}:
            shapes.append(shape)
    return shapes


def _build_hole_layout(side: int, k_target: int):
    occ = np.ones((side, side), dtype=bool)
    shape_records: list[dict] = []

    for shape in _choose_shapes(side, k_target):
        sh, sw = shape.shape
        anchors = [(r, c) for r in range(1, side - sh) for c in range(1, side - sw)]
        random.shuffle(anchors)
        placed_orbit = None
        for r0, c0 in anchors[:PLACEMENT_RETRIES]:
            orbit = _shape_orbit(shape, r0, c0, side)
            if orbit is None:
                continue
            if _orbit_fits(occ, orbit):
                _carve_orbit(occ, orbit)
                placed_orbit = orbit
                break
        if placed_orbit is None:
            return None

        placements = []
        for coords in placed_orbit:
            (pr, pc), mask = _coords_to_mask(coords)
            placements.append((int(pr), int(pc), mask))
        shape_records.append(
            {
                "shape": np.array(shape, dtype=bool),
                "key": _canonical_key(shape),
                "placements": placements,
            }
        )

    hole_mask = ~occ
    if not _hole_mask_is_valid(hole_mask):
        return None

    frame_density = float(np.mean(occ))
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(rec["placements"]) for rec in shape_records)
    if total_holes < 12:
        return None

    return occ, shape_records


def _build_output(occ: np.ndarray, shape_records: list[dict], frame_col: int, bg: int):
    side = int(occ.shape[0])
    output = np.full((side, side), int(frame_col), dtype=int)
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(shape_records):
        return None
    shape_colors = random.sample(avail_colors, len(shape_records))

    shape_info: list[dict] = []
    for rec, fill_col in zip(shape_records, shape_colors):
        for pr, pc, mask in rec["placements"]:
            patch = np.full(mask.shape, int(fill_col), dtype=int)
            paste(output, patch, int(pr), int(pc), mask=mask)
        shape_info.append(
            {
                "shape": np.array(rec["shape"], dtype=bool),
                "key": rec["key"],
                "fill_color": int(fill_col),
                "placements": rec["placements"],
            }
        )
    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, ori: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(ori.shape, int(bg), dtype=int)
    patch[ori] = int(color)
    try:
        r0, c0 = random_free_location(placement, patch, transparent=int(bg), padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, int(r0), int(c0), transparent=int(bg))
    paste(inp, patch, int(r0), int(c0), transparent=int(bg))
    return True


def _scatter_sprites(inp: np.ndarray, shape_info: list[dict], bg: int, frame_col: int) -> bool:
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for rec in shape_info:
        fill_color = int(rec["fill_color"])
        orientations = _all_orientations(rec["shape"])
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        if not losers:
            return False
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers)))
        losing_colors = random.sample(losers, n_other)

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        for lose_col in losing_colors:
            n_lose = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_lose):
                if not _place_sprite(inp, placement, random.choice(orientations), int(lose_col), bg):
                    break
    return True


def _analyze_generated_input(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    frame = max(comps, key=lambda comp: int(comp["area"]))
    frame_col = int(frame["color"])
    sub = crop_bbox(g, frame["bbox"])
    hole_mask = sub != frame_col
    hole_shapes = []
    for coords in components_mask(hole_mask, connectivity=4):
        _, mask = _coords_to_mask(coords)
        hole_shapes.append(mask)
    return {
        "frame_side": int(sub.shape[0]),
        "frame_square": bool(sub.shape[0] == sub.shape[1]),
        "hole_count": int(len(hole_shapes)),
        "rot90_layout": bool(np.array_equal(hole_mask, np.rot90(hole_mask, 1))),
        "lr_layout": bool(np.array_equal(hole_mask, np.fliplr(hole_mask))),
        "ud_layout": bool(np.array_equal(hole_mask, np.flipud(hole_mask))),
        "all_holes_rot_sym": bool(all(_shape_has_rot_sym(mask) for mask in hole_shapes)),
        "special_hole_count": int(sum(1 for mask in hole_shapes if _shape_has_rot90(mask) and int(np.sum(mask)) not in {1, 4})),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    k_target = random.choices(SHAPE_COUNT_OPTIONS, weights=SHAPE_COUNT_WEIGHTS, k=1)[0]
    if k_target >= 4:
        side_low = max(FRAME_SIDE_RANGE[0], FOUR_SHAPE_MIN_SIDE)
        side = random.randint(side_low, FRAME_SIDE_RANGE[1])
    else:
        side = random.randint(*FRAME_SIDE_RANGE)

    layout = _build_hole_layout(side, k_target)
    if layout is None:
        return None
    occ, shape_records = layout

    built = _build_output(occ, shape_records, frame_col, bg)
    if built is None:
        return None
    output, shape_info = built

    fr = random.randint(0, GRID_SIZE - side)
    fc = random.randint(0, GRID_SIZE - side)
    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    frame_patch = np.where(occ, int(frame_col), int(bg)).astype(int)
    paste(inp, frame_patch, int(fr), int(fc))

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
        return None

    metrics = _analyze_generated_input(inp)
    if not (metrics["frame_square"] and metrics["rot90_layout"] and not metrics["lr_layout"] and not metrics["ud_layout"] and metrics["all_holes_rot_sym"]):
        return None
    if metrics["special_hole_count"] < 4:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid 67e490f4 combo D7/2+D8/2 input after {MAX_GEN_ATTEMPTS} attempts")
