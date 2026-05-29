# Combo variant of 67e490f4 (index 27)
# combo_spec: D1/2+D11/0+D12/1
# Name: 面积匹配+1:1+无干扰

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
    random_free_location,
)


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - Original task: hole shape -> gather all outside sprites with the same canonical
#      shape -> use the plurality color for that shape.
#    - Fused variant: hole area -> read the single outside sprite for that area -> use
#      that sprite's color directly. Example:
#        inside holes: area-3 bar, area-3 L, area-4 square, area-5 T
#        outside sprites: one red area-3 sprite, one green area-4 sprite, one yellow
#        area-5 sprite; no extra objects.
#      Original logic would compare masks and fail to explain why both area-3 holes
#      become red; the fused logic counts cells and lets both area-3 holes share the
#      same color from the only area-3 sprite.
# b) Complexity calibration
#    - The generator forces at least one area class to contain >=2 distinct hole
#      shapes. Otherwise area matching would visually collapse back to shape matching.
#    - Only 2-4 area classes and 4-6 total holes are used, so samples stay readable
#      and every active rule gets repeated support.
# c) Hidden-assumption audit
#    - Original `key = canonical_shape(mask)` is invalid because multiple different
#      hole shapes intentionally merge into one area class.
#    - Original `color -> count` voting is removed: D11/0 + D12/1 mean one clean clue
#      per active area, so the solver must assert a unique sprite/color per hole area.
#    - Interpreting D11/0 at the old shape level would conflict with D1/2. The fusion
#      therefore applies `1:1` to the current matching class: one sprite per used area.
# d) Placement legality
#    - Holes are placed fully inside a solid rectangular frame with a one-cell border.
#    - Holes are 4-disconnected, sprites are fully outside the frame, and
#      `random_free_location(..., padding=1)` preserves separation.
# e) Visual observability
#    - Every sample has a witness area with multiple different hole shapes but only one
#      outside sprite. A human can see that shape is no longer the decisive key.
#    - Because there are no losing colors and no distractor areas, the remaining task is
#      plainly "count cells, then copy the unique color for that count".
# f) Visual cleanliness
#    - Outside clutter is minimized to one monochrome sprite per active area class.
#    - Unique colors per area keep the mapping readable even when two hole shapes share
#      the same fill.
# g) Color inferability
#    - Colors are assigned one-per-area from the visible palette excluding bg/frame.
#      There is no invisible tiebreak or color-id arithmetic.
# h) Generator-side guarantees for new solver assumptions
#    - Every hole area appears in exactly one outside sprite.
#    - No sprite area is unused by the holes.
#    - At least one hole canonical shape has no exact sprite-shape mate, proving that
#      the generated sample really needs area matching rather than shape matching.


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    base = mask.astype(bool)
    for flip in (False, True):
        flipped = np.flipud(base) if flip else base
        for k in range(4):
            oriented = np.rot90(flipped, k)
            yield (oriented.shape, oriented.tobytes()), oriented


def _canonical_key(mask: np.ndarray):
    """Canonical representative for a shape under rotation/flip."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _shape_area(mask: np.ndarray) -> int:
    return int(np.sum(mask))


def _mask_from_coords(coords: np.ndarray):
    """Return a tight boolean mask and top-left for a component."""
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0))


def _find_frame(grid: np.ndarray, bg: int):
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    return int(frame["color"]), tuple(map(int, frame["bbox"]))


def _extract_holes(sub: np.ndarray, frame_col: int):
    holes = []
    for coords in components_mask(sub != int(frame_col), connectivity=4):
        mask, (r0, c0) = _mask_from_coords(coords)
        holes.append(
            {
                "area": int(coords.shape[0]),
                "key": _canonical_key(mask),
                "pos": (int(r0), int(c0)),
                "mask": mask,
            }
        )
    return holes


def _extract_sprites(grid: np.ndarray, bg: int, frame_bbox):
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = grid.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    sprites = []
    for comp in comps:
        mask, _ = _mask_from_coords(comp["coords"])
        sprites.append(
            {
                "area": int(comp["area"]),
                "color": int(comp["color"]),
                "key": _canonical_key(mask),
                "mask": mask,
            }
        )
    return sprites


def analyze_fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether generated samples genuinely realize the fused rule.

    Key metrics:
    - shared_area_count >= 1 means multiple hole shapes share an area class.
    - repeated_sprite_area_count == 0 and extra/missing area counts == 0 mean D11/0+D12/1
      are active at the area level.
    - unmatched_hole_shape_count_under_shape_rule >= 1 means shape matching would not
      solve the same input, so the sample does not degenerate back to the original task.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_col, frame_bbox = _find_frame(g, bg)
    holes = _extract_holes(crop_bbox(g, frame_bbox), frame_col)
    sprites = _extract_sprites(g, bg, frame_bbox)

    hole_areas = {int(hole["area"]) for hole in holes}
    sprite_areas = {int(sprite["area"]) for sprite in sprites}
    hole_keys = {hole["key"] for hole in holes}
    sprite_keys = {sprite["key"] for sprite in sprites}

    shapes_per_area: dict[int, set] = {}
    for hole in holes:
        shapes_per_area.setdefault(int(hole["area"]), set()).add(hole["key"])

    sprites_per_area: dict[int, int] = {}
    colors_per_area: dict[int, set[int]] = {}
    for sprite in sprites:
        area = int(sprite["area"])
        sprites_per_area[area] = int(sprites_per_area.get(area, 0)) + 1
        colors_per_area.setdefault(area, set()).add(int(sprite["color"]))

    return {
        "hole_count": int(len(holes)),
        "sprite_count": int(len(sprites)),
        "used_area_count": int(len(hole_areas)),
        "shared_area_count": int(sum(1 for keys in shapes_per_area.values() if len(keys) >= 2)),
        "max_shapes_per_area": int(max((len(keys) for keys in shapes_per_area.values()), default=0)),
        "repeated_sprite_area_count": int(sum(1 for count in sprites_per_area.values() if count != 1)),
        "extra_sprite_area_count": int(len(sprite_areas - hole_areas)),
        "missing_sprite_area_count": int(len(hole_areas - sprite_areas)),
        "multicolor_area_count": int(sum(1 for colors in colors_per_area.values() if len(colors) > 1)),
        "unmatched_hole_shape_count_under_shape_rule": int(sum(1 for key in hole_keys if key not in sprite_keys)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame by coloring each hole from the unique outside sprite with the same area.

    Input structure:
    - The background is the most common color.
    - One large monochrome rectangular frame contains several small holes of different shapes.
    - Outside the frame, each active hole area has exactly one monochrome sprite and there are no irrelevant sprites.

    Rule:
    - Find the frame, crop to its bounding box, and start from a solid block of the frame color.
    - For each interior hole, count its filled cells.
    - Fill that hole with the color of the unique outside sprite whose pixel area matches the hole area.

    Key insight:
    - The outside objects are area keys, not shape keys: different hole shapes with the same cell count share one external clue and therefore receive the same color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_col, frame_bbox = _find_frame(g, bg)

    sub = crop_bbox(g, frame_bbox)
    th, tw = sub.shape
    holes = _extract_holes(sub, frame_col)
    sprites = _extract_sprites(g, bg, frame_bbox)

    area_to_color: dict[int, int] = {}
    for sprite in sprites:
        area = int(sprite["area"])
        color = int(sprite["color"])
        prev = area_to_color.get(area)
        assert prev is None or prev == color
        area_to_color[area] = color

    hole_areas = {int(hole["area"]) for hole in holes}
    assert set(area_to_color) == hole_areas

    out = np.full((th, tw), int(frame_col), dtype=int)
    for hole in holes:
        area = int(hole["area"])
        hr0, hc0 = hole["pos"]
        mask = hole["mask"]
        patch = np.full(mask.shape, int(area_to_color[area]), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=mask)
    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (8, 13)
FRAME_W_RANGE = (9, 18)

TOTAL_SHAPE_COUNT_RANGE = (4, 6)
AREA_GROUP_COUNT_RANGE = (2, 4)
MIN_SHARED_AREA_SHAPES = 2
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.70, 0.92)
PLACEMENT_RETRIES = 140
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 300


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [1, 0]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1], [0, 1], [0, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[0, 1], [1, 1], [1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 1], [0, 1], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    library: list[np.ndarray] = []
    for spec in raw:
        mask = np.array(spec, dtype=bool)
        if _shape_area(mask) > SHAPE_MAX_AREA:
            continue
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            library.append(mask)
    return library


SHAPE_LIBRARY = _build_shape_library()


def _shapes_by_area(th: int, tw: int):
    groups: dict[int, list[np.ndarray]] = {}
    for shape in SHAPE_LIBRARY:
        if shape.shape[0] > th - 2 or shape.shape[1] > tw - 2:
            continue
        area = _shape_area(shape)
        groups.setdefault(area, []).append(shape)
    return groups


def _can_place(occ: np.ndarray, shape: np.ndarray, pr: int, pc: int) -> bool:
    """Check bounds, overlap, and 4-neighbor separation for one hole."""
    th, tw = occ.shape
    sh, sw = shape.shape
    if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
        return False

    new_cells = set()
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                new_cells.add((pr + dr, pc + dc))

    for rr, cc in new_cells:
        if not occ[rr, cc]:
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                return False
    return True


def _carve_hole(occ: np.ndarray, shape: np.ndarray, pr: int, pc: int):
    sh, sw = shape.shape
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                occ[pr + dr, pc + dc] = False


def _select_shapes(th: int, tw: int):
    by_area = _shapes_by_area(th, tw)
    candidate_areas = [area for area, shapes in by_area.items() if shapes]
    shared_areas = [area for area, shapes in by_area.items() if len(shapes) >= MIN_SHARED_AREA_SHAPES]
    if len(candidate_areas) < AREA_GROUP_COUNT_RANGE[0] or not shared_areas:
        return None

    n_areas = random.randint(AREA_GROUP_COUNT_RANGE[0], min(AREA_GROUP_COUNT_RANGE[1], len(candidate_areas)))
    witness_area = random.choice(shared_areas)
    other_areas = [area for area in candidate_areas if area != witness_area]
    chosen_areas = [witness_area]
    if n_areas > 1:
        chosen_areas += random.sample(other_areas, n_areas - 1)

    counts = {area: 1 for area in chosen_areas}
    counts[witness_area] = 2

    area_caps = {area: len(by_area[area]) for area in chosen_areas}
    min_total = max(TOTAL_SHAPE_COUNT_RANGE[0], sum(counts.values()))
    max_total = min(TOTAL_SHAPE_COUNT_RANGE[1], sum(area_caps.values()))
    if min_total > max_total:
        return None

    target_total = random.randint(min_total, max_total)
    while sum(counts.values()) < target_total:
        expandable = [area for area in chosen_areas if counts[area] < area_caps[area]]
        if not expandable:
            break
        counts[random.choice(expandable)] += 1

    selected = []
    for area in chosen_areas:
        pool = list(by_area[area])
        random.shuffle(pool)
        selected.extend(pool[: counts[area]])

    random.shuffle(selected)
    selected.sort(key=lambda shape: (_shape_area(shape), shape.shape[0] * shape.shape[1]), reverse=True)
    return selected


def _build_hole_layout(th: int, tw: int):
    """
    Place one instance of each selected shape.
    Returns (occ, hole_records) where each record stores the base shape, its oriented hole
    mask, position, and area.
    """
    selected = _select_shapes(th, tw)
    if selected is None:
        return None

    occ = np.ones((th, tw), dtype=bool)
    hole_records: list[dict[str, object]] = []

    for base_shape in selected:
        orientations = _all_orientations(base_shape)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            oriented = random.choice(orientations)
            sh, sw = oriented.shape
            pr = random.randint(1, th - sh - 1)
            pc = random.randint(1, tw - sw - 1)
            if not _can_place(occ, oriented, pr, pc):
                continue
            _carve_hole(occ, oriented, pr, pc)
            hole_records.append(
                {
                    "base_shape": base_shape,
                    "hole_mask": oriented,
                    "hole_pos": (int(pr), int(pc)),
                    "area": _shape_area(base_shape),
                    "key": _canonical_key(base_shape),
                }
            )
            placed = True
            break
        if not placed:
            return None

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    shapes_per_area: dict[int, set] = {}
    for record in hole_records:
        area = int(record["area"])
        shapes_per_area.setdefault(area, set()).add(record["key"])
    if max((len(keys) for keys in shapes_per_area.values()), default=0) < MIN_SHARED_AREA_SHAPES:
        return None

    return occ, hole_records


def _build_output(occ: np.ndarray, hole_records: list[dict[str, object]], frame_col: int, bg: int):
    used_areas = sorted({int(record["area"]) for record in hole_records})
    avail_colors = [color for color in range(NUM_COLORS) if color not in (bg, frame_col)]
    if len(avail_colors) < len(used_areas):
        return None

    area_colors = dict(zip(used_areas, random.sample(avail_colors, len(used_areas))))
    output = np.full(occ.shape, frame_col, dtype=int)
    shape_info: list[dict[str, object]] = []

    for record in hole_records:
        hole_mask = record["hole_mask"]
        pr, pc = record["hole_pos"]
        area = int(record["area"])
        fill_color = int(area_colors[area])
        patch = np.full(hole_mask.shape, fill_color, dtype=int)
        paste(output, patch, int(pr), int(pc), mask=hole_mask)
        shape_info.append(
            {
                "base_shape": record["base_shape"],
                "hole_mask": hole_mask,
                "hole_pos": (int(pr), int(pc)),
                "area": area,
                "key": record["key"],
                "fill_color": fill_color,
            }
        )

    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, oriented: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(oriented.shape, bg, dtype=int)
    patch[oriented] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info: list[dict[str, object]], bg: int) -> bool:
    """
    Place exactly one sprite for each used hole area.
    The chosen sprite shape is one of the hole shapes from that area, so every outside
    clue remains relevant while same-area sibling hole shapes still require area matching.
    """
    placement = inp.copy()
    by_area: dict[int, list[dict[str, object]]] = {}
    for info in shape_info:
        by_area.setdefault(int(info["area"]), []).append(info)

    for area_infos in by_area.values():
        rep = random.choice(area_infos)
        base_shape = rep["base_shape"]
        hole_mask = rep["hole_mask"]
        fill_color = int(rep["fill_color"])
        orientations = _all_orientations(base_shape)
        hole_signature = (hole_mask.shape, hole_mask.tobytes())
        diff_orientations = [ori for ori in orientations if (ori.shape, ori.tobytes()) != hole_signature]
        oriented = random.choice(diff_orientations or orientations)
        if not _place_sprite(inp, placement, oriented, fill_color, bg):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([color for color in range(NUM_COLORS) if color != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_records = layout

    result = _build_output(occ, hole_records, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    if not _scatter_sprites(inp, shape_info, bg):
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
        metrics = analyze_fusion_metrics(inp)
        if (
            metrics["shared_area_count"] >= 1
            and metrics["repeated_sprite_area_count"] == 0
            and metrics["extra_sprite_area_count"] == 0
            and metrics["missing_sprite_area_count"] == 0
            and metrics["multicolor_area_count"] == 0
            and metrics["unmatched_hole_shape_count_under_shape_rule"] >= 1
            and metrics["sprite_count"] == metrics["used_area_count"] < metrics["hole_count"]
        ):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
