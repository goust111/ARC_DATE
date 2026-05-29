# Combo variant of 67e490f4 (index 2)
# combo_spec: D1/2+D2/0
# Name: 面积匹配+禁止旋转

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


# Design analysis
# ---------------
# a) Concrete behavior trace:
#    - Original task: each hole inside the frame was keyed by its shape up to rotation/flip,
#      and outside sprites with that canonical shape voted on the fill color.
#    - Combo task: the only matching key is pixel area. A 4-cell zigzag hole, a 4-cell L hole,
#      and a rotated 4-cell L hole all belong to the same bucket because they each contain
#      four foreground cells. Sprites are counted once exactly as drawn; we never expand a
#      sprite into rotated/flipped aliases.
#    - Small trace:
#        holes inside frame:
#          H1 = area-4 L
#          H2 = area-4 rotated L
#          H3 = area-4 zigzag
#        outside sprites:
#          L(area 4, blue) x3
#          rotated-L(area 4, blue) x3
#          zigzag(area 4, red) x4
#          zigzag(area 4, blue) x1
#      Original rule: H1/H2 would follow the L family, H3 would follow zigzag, so H3 -> red.
#      Combo rule: all area-4 sprites vote together, blue wins 7 to 4, so H1/H2/H3 -> blue.
#      The D2 part still matters operationally: each sprite contributes only in its native
#      orientation, and generation guarantees every hole orientation has exact-orientation
#      witnesses outside the frame instead of relying on hidden mental rotation.
# b) Complexity calibration:
#    - If every used area had only one hole shape, the combo would visually collapse toward the
#      source task. The generator therefore forces one witness area containing three exact masks:
#      two orientations from one asymmetric family plus one different family of the same area.
#    - Exactly 3 areas are active per grid, which keeps the evidence readable while avoiding
#      lopsided coverage between a 2-area and 3-area generator branch.
# c) Implicit assumption audit against the old solver:
#    - Step 1 (largest monochrome non-bg component is the frame) still holds.
#    - Old Step 2 stored `(mask.shape, mask.tobytes())` as the hole key. That assumption is
#      invalid now; multiple unrelated masks may intentionally share one color because their
#      areas are equal, so the key becomes `pixel_count`.
#    - Old Step 3 expanded every sprite into 8 rotated/flipped keys. That premise is also false:
#      under the combo we do no orientation normalization at all. A sprite is observed only in
#      the orientation it is drawn in, then its area is counted once.
#    - Step 4 assumes each used area has at least one outside witness and a readable plurality.
#      The generator enforces both, and for the witness area it also enforces disagreement with
#      the original shape-matching rule.
# d) Placement legality:
#    - Hole carving still keeps every hole strictly inside the frame border and 4-disconnected
#      from all other holes.
#    - Sprite scattering still uses `random_free_location(..., padding=1)`, so sprites remain
#      separated from one another and from the frame bbox.
# e) Visual observability:
#    - Every sample contains one area class shared by multiple exact masks. Two of those masks
#      are visibly different orientations of the same family, and another is a different family
#      of the same area. The output colors all of them identically, making area the only stable
#      explanation.
#    - For that witness area, one exact shape has a local plurality in a losing color, but the
#      global area vote still selects the final color. This guarantees visible divergence from
#      the original task instead of a silent degeneration.
# f) Visual cleanliness:
#    - The scene layout stays close to the source task: one frame, sparse non-touching sprites,
#      and only a small number of active areas/colors.
# g) Color inferability:
#    - Colors are assigned per used area, not per exact shape. Each area gets a strict winner,
#      so humans never need invisible color-id tie breaks.
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes area support exists for every hole and that the winning color is unique
#      enough to read. `generate_input()` guarantees both.
#    - `main()` no longer distinguishes shape orientation, so generator must carry the D2 signal
#      by construction: every hole orientation is shown with exact-orientation outside sprites,
#      and no winning evidence depends on rotating a sprite mentally.


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = mask.astype(bool)
    return (m.shape, m.tobytes())


def _all_rot_flip(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield _mask_key(m2), m2


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _shape_area(mask: np.ndarray) -> int:
    return int(np.sum(mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the holes of a large frame by counting outside sprites with the same area.

    Input structure:
    - The most common color is the background, with one large monochrome frame object.
    - Inside the frame bbox are several small holes cut out of the frame color.
    - Outside the frame bbox are small sprites scattered on the background.

    Rule:
    - Crop to the frame's bounding box and begin with a solid rectangle of the frame color.
    - For each hole, count its foreground pixels and gather every outside sprite with that same area.
    - Fill the hole with the most frequent color among those same-area sprites; sprites are counted
      only as drawn, with no rotation or flip normalization.

    Key insight:
    - The outside pieces are area-coded votes rather than shape-coded keys. Orientation stays literal
      in the evidence, but the output decision ignores shape and direction and depends only on area.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the unique template frame: the largest monochrome non-bg component.
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])

    sub = crop_bbox(g, frame["bbox"])
    th, tw = sub.shape

    # STEP 2) Extract hole masks inside the frame, but key them only by pixel count.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[int, tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((int(coords.shape[0]), (int(hr0), int(hc0)), mask))

    # STEP 3) Extract outside sprites and count each one once in its native orientation.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_area: dict[int, dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        area = int(coords.shape[0])
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        bucket = sprites_by_area.setdefault(area, {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Fill each hole from the same-area color vote.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for area, (hr0, hc0), hmask in holes:
        cmap = sprites_by_area.get(int(area))
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        paste(out, np.full(hmask.shape, int(scol), dtype=int), int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

AREA_GROUP_COUNT_RANGE = (3, 3)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 90
LAYOUT_SELECTION_RETRIES = 24
MULT_EVEN_WEIGHTS = ([1, 2], [5, 1])
MULT_ODD_WEIGHTS = ([1, 2], [5, 1])

WINNING_SPRITE_RANGE = (2, 3)
WITNESS_LOSER_BONUS_RANGE = (1, 2)
OTHER_COLOR_MAX = 1
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 1]],
        [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 0, 1], [1, 1, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for arr in raw:
        mask = np.array(arr, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen and _shape_area(mask) <= SHAPE_MAX_AREA:
            seen.add(key)
            result.append(mask)
    return result


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int) -> list[tuple[int, int]]:
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


def _validate_positions(
    occ: np.ndarray,
    shape: np.ndarray,
    positions: list[tuple[int, int]],
    th: int,
    tw: int,
) -> list[tuple[int, int]] | None:
    sh, sw = shape.shape
    new_cells: set[tuple[int, int]] = set()
    owners: dict[tuple[int, int], int] = {}
    for idx, (pr, pc) in enumerate(positions):
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    cell = (pr + dr, pc + dc)
                    new_cells.add(cell)
                    owners[cell] = idx

    for (rr, cc), owner in owners.items():
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if (nr, nc) in owners and owners[(nr, nc)] != owner:
                return None

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
    r_max, c_max = th // 2 - sh, tw // 2 - sw
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
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ: np.ndarray, shape: np.ndarray, positions: list[tuple[int, int]]) -> None:
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _families_by_area(max_sh: int, max_sw: int) -> dict[int, list[dict[str, object]]]:
    groups: dict[int, list[dict[str, object]]] = {}
    for base in SHAPE_LIBRARY:
        orientations = [
            oriented
            for oriented in _all_orientations(base)
            if oriented.shape[0] <= max_sh and oriented.shape[1] <= max_sw and _shape_area(oriented) <= SHAPE_MAX_AREA
        ]
        if not orientations:
            continue
        area = _shape_area(base)
        groups.setdefault(area, []).append(
            {
                "family": _canonical_key(base),
                "orientations": orientations,
            }
        )
    return groups


def _select_oriented_groups(th: int, tw: int) -> list[dict[str, object]] | None:
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)
    by_area = _families_by_area(max_sh, max_sw)
    if len(by_area) < AREA_GROUP_COUNT_RANGE[0]:
        return None

    witness_candidates = []
    for area, families in by_area.items():
        paired = [fam for fam in families if len(fam["orientations"]) >= 2]
        if paired and len(families) >= 2:
            witness_candidates.append((area, paired, families))
    if not witness_candidates:
        return None

    witness_area, paired_families, witness_families = random.choice(witness_candidates)
    paired_family = random.choice(paired_families)
    contrast_family = random.choice([fam for fam in witness_families if fam["family"] != paired_family["family"]])
    ori1, ori2 = random.sample(list(paired_family["orientations"]), 2)

    groups = [
        {"area": witness_area, "family": paired_family["family"], "shape": ori1, "role": "paired"},
        {"area": witness_area, "family": paired_family["family"], "shape": ori2, "role": "paired"},
        {
            "area": witness_area,
            "family": contrast_family["family"],
            "shape": random.choice(list(contrast_family["orientations"])),
            "role": "contrast",
        },
    ]

    other_areas = [area for area in by_area if area != witness_area]
    area_count = random.randint(AREA_GROUP_COUNT_RANGE[0], min(AREA_GROUP_COUNT_RANGE[1], len(by_area)))
    chosen_other_areas = random.sample(other_areas, area_count - 1)
    for area in chosen_other_areas:
        family = random.choice(by_area[area])
        groups.append(
            {
                "area": area,
                "family": family["family"],
                "shape": random.choice(list(family["orientations"])),
                "role": "regular",
            }
        )

    return groups


def _build_hole_layout(th: int, tw: int):
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if (th % 2 == 0 or tw % 2 == 0) else MULT_ODD_WEIGHTS

    for _ in range(LAYOUT_SELECTION_RETRIES):
        selected_groups = _select_oriented_groups(th, tw)
        if selected_groups is None or len(selected_groups) < MIN_SHAPES:
            continue

        selected_groups = sorted(
            selected_groups,
            key=lambda group: (-_shape_area(np.array(group["shape"], dtype=bool)), -np.array(group["shape"]).shape[0]),
        )

        occ = np.ones((th, tw), dtype=bool)
        placed_groups: list[dict[str, object]] = []
        failed = False

        for group in selected_groups:
            shape = np.array(group["shape"], dtype=bool)
            mult = random.choices(mult_vals, weights=mult_wts)[0]
            placements = []
            for _ in range(mult):
                for _ in range(PLACEMENT_RETRIES):
                    positions = _try_place_symmetric(occ, shape, th, tw)
                    if positions is not None:
                        _carve_hole(occ, shape, positions)
                        placements.append(positions)
                        break
            if not placements:
                failed = True
                break
            placed_groups.append(
                {
                    "area": int(group["area"]),
                    "family": group["family"],
                    "shape": shape,
                    "role": str(group["role"]),
                    "placements": placements,
                }
            )

        if failed:
            continue

        frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
        if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
            continue

        total_holes = sum(len(pos) for group in placed_groups for pos in group["placements"])
        if total_holes < MIN_TOTAL_HOLES:
            continue

        witness_groups = [group for group in placed_groups if group["role"] in {"paired", "contrast"}]
        if len(witness_groups) < 3:
            continue

        return occ, placed_groups

    return None


def _build_output(occ: np.ndarray, placed_groups: list[dict[str, object]], frame_col: int, bg: int):
    th, tw = occ.shape
    used_areas = sorted({int(group["area"]) for group in placed_groups})
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(used_areas):
        return None

    area_colors = dict(zip(used_areas, random.sample(avail_colors, len(used_areas))))
    output = np.full((th, tw), frame_col, dtype=int)
    shape_info: list[dict[str, object]] = []

    for group in placed_groups:
        shape = np.array(group["shape"], dtype=bool)
        fill_color = int(area_colors[int(group["area"])])
        all_pos: list[tuple[int, int]] = []
        for positions in group["placements"]:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_color, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append(
            {
                "area": int(group["area"]),
                "family": group["family"],
                "shape": shape,
                "role": str(group["role"]),
                "fill_color": fill_color,
                "positions": all_pos,
            }
        )

    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, shape: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _place_many(
    inp: np.ndarray,
    placement: np.ndarray,
    shape: np.ndarray,
    color: int,
    count: int,
    bg: int,
    must_succeed: bool,
) -> bool:
    for _ in range(count):
        ok = _place_sprite(inp, placement, shape, color, bg)
        if not ok and must_succeed:
            return False
        if not ok:
            break
    return True


def _scatter_sprites(
    inp: np.ndarray,
    shape_info: list[dict[str, object]],
    bg: int,
    frame_col: int,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    placement = inp.copy()
    r0, c0, r1, c1 = frame_bbox
    placement[r0 : r1 + 1, c0 : c1 + 1] = int(frame_col)
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    by_area: dict[int, list[dict[str, object]]] = {}
    for info in shape_info:
        by_area.setdefault(int(info["area"]), []).append(info)

    witness_areas = [area for area, infos in by_area.items() if any(info["role"] == "contrast" for info in infos)]
    assert len(witness_areas) == 1
    witness_area = witness_areas[0]

    for area, infos in by_area.items():
        fill_color = int(infos[0]["fill_color"])
        if area == witness_area:
            paired_infos = [info for info in infos if info["role"] == "paired"]
            contrast_infos = [info for info in infos if info["role"] == "contrast"]
            if len(paired_infos) < 2 or len(contrast_infos) != 1:
                return False

            contrast_info = contrast_infos[0]
            loser_choices = [c for c in other_pool if c != fill_color]
            if not loser_choices:
                return False
            loser_color = random.choice(loser_choices)

            fill_total = 0
            for info in paired_infos:
                n_win = random.randint(*WINNING_SPRITE_RANGE)
                fill_total += n_win
                if not _place_many(inp, placement, info["shape"], fill_color, n_win, bg, must_succeed=True):
                    return False

            contrast_win = 1
            contrast_lose = fill_total - 2
            contrast_lose = max(contrast_win + random.randint(*WITNESS_LOSER_BONUS_RANGE), contrast_lose)
            if contrast_lose >= fill_total + contrast_win:
                contrast_lose = fill_total + contrast_win - 1
            fill_total += contrast_win

            if not _place_many(inp, placement, contrast_info["shape"], fill_color, contrast_win, bg, must_succeed=True):
                return False
            if not _place_many(inp, placement, contrast_info["shape"], loser_color, contrast_lose, bg, must_succeed=True):
                return False
        else:
            for info in infos:
                n_win = random.randint(*WINNING_SPRITE_RANGE)
                if not _place_many(inp, placement, info["shape"], fill_color, n_win, bg, must_succeed=True):
                    return False

                losers = [c for c in other_pool if c != fill_color]
                n_other = random.randint(0, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
                for oc in random.sample(losers, n_other):
                    n_oc = random.randint(1, max(1, n_win - 1))
                    if not _place_many(inp, placement, info["shape"], oc, n_oc, bg, must_succeed=False):
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
    occ, placed_groups = layout

    built = _build_output(occ, placed_groups, frame_col, bg)
    if built is None:
        return None
    output, shape_info = built

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    if not _scatter_sprites(inp, shape_info, bg, frame_col, (fr, fc, fr + th - 1, fc + tw - 1)):
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
