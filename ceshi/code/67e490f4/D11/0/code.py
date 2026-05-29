# Variant of 67e490f4
# Dimension: D11 — 精灵数量与洞的对应关系 (meta)
# Change: many:1 — 每种洞形状对应多个精灵（2-13个，多种颜色），通过频率消歧选择填充色... → 1:1 精确对应

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

# Design analysis for the D11 variant
# -----------------------------------
# a) Concrete behavior and trace
#    - The input still has one large monochrome frame, several holes inside it, and
#      several small sprites outside it on the background.
#    - The D11 change is localized to the mapping between hole shapes and sprites:
#      the original task allowed many sprites with the same canonical shape and used
#      color frequency to decide the fill; this variant keeps exactly one outside
#      sprite per canonical hole shape, so the sprite color is read directly.
#    - Example trace:
#      original: two L-shaped red sprites + one L-shaped green sprite outside ->
#      an L-shaped hole is filled red by plurality.
#      variant: one L-shaped green sprite outside -> the L-shaped hole is filled
#      green immediately; there is no vote and no tie-breaking.
# b) Complexity calibration
#    - If each hole shape appeared many times inside the frame, a single outside
#      sprite could still solve the grid, but the D11 change would be visually weak.
#      To make the one-to-one mapping obvious, the generator uses 4-6 holes and each
#      canonical shape appears exactly once in the frame and once outside it.
#    - To keep the task non-trivial, shapes may still be rotated or flipped between
#      hole and sprite, and the library includes asymmetric polyominoes.
# c) Audit of original solver assumptions that depended on the old D11
#    - The original `sprites_by_key: key -> color -> count` assumed duplicates are
#      expected and that the correct color comes from `max(count)`. That is no longer
#      the rule and would silently preserve the old behavior.
#    - The original hole extraction and frame detection do not rely on multiplicity,
#      so they can stay with only key normalization adjusted.
#    - The original outside-sprite scan tolerated multi-color connected components.
#      This variant generates monochrome sprites only, so the solver now reads one
#      sprite color per canonical shape directly.
# d) Placement legality
#    - Every hole is placed fully inside the frame with a one-cell frame border.
#    - Holes are kept 4-disconnected so each one remains an independent component.
#    - Outside sprites are placed with `random_free_location(..., padding=1)`, which
#      keeps them separated from one another and from the frame by background cells.
# e) Visual observability
#    - The generated examples visibly show that the number of outside sprites equals
#      the number of holes, unlike the original many-sprites-per-shape regime.
#    - At least one chosen shape normally supports multiple orientations, and the
#      generator tries to orient one sprite differently from its matching hole so the
#      shape-only correspondence remains visible.
# f) Visual cleanliness
#    - There are no distractor sprites and no repeated shape groups, so the outside
#      region is much cleaner than the original task and each pairing is easy to see.
#    - Unique fill colors are assigned to hole shapes so different matches do not
#      visually collapse into the same outcome.
# g) Color-change inferability
#    - Colors are not transformed or sorted. A hole simply inherits the color of its
#      unique matching sprite. Distinct colors across shapes make this rule readable
#      from a few examples.
#
# New solver assumptions and how the generator guarantees them
#    - Largest monochrome non-background component is the frame:
#      frame dimensions are much larger than any sprite.
#    - Every non-frame connected component inside the frame bbox is a hole:
#      generator only carves background holes inside the frame and adds no interior
#      decorations.
#    - Each canonical outside shape maps to exactly one color:
#      generator chooses unique canonical shapes and scatters exactly one sprite for
#      each chosen shape, with no distractors.

# --------------- solver ---------------


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame by pairing each interior hole with its single matching outside sprite.

    Input structure:
    - The background is the most common color.
    - One large monochrome component is the frame; inside its bounding box are several hole components.
    - Outside the frame there is exactly one small monochrome sprite for each hole shape, possibly rotated or flipped.

    Rule:
    - Find the frame, crop to its bounding box, and start a solid canvas of the frame color.
    - Extract each hole shape inside the crop and normalize it up to rotation/flip.
    - Match that normalized shape to the unique outside sprite with the same normalized shape and paint the hole with that sprite's color.

    Key insight:
    - In this variant the outside sprites form a one-to-one shape-to-color dictionary, so once the shape matches, the fill color is determined directly with no frequency vote.
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

    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((_canonical_key(mask), (int(hr0), int(hc0)), mask))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        key = _canonical_key(mask)
        scol = int(comp["color"])
        prev = sprites_by_key.get(key)
        assert prev is None or prev == scol
        sprites_by_key[key] = scol

    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), mask in holes:
        scol = sprites_by_key.get(key)
        assert scol is not None
        patch = np.full(mask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=mask)

    return out


# --------------- generator ---------------

# ── Configuration ──────────────────────────────────────────────────────
NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 12)
FRAME_W_RANGE = (8, 18)

HOLE_COUNT_RANGE = (4, 6)
MIN_HOLES = 4
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.70, 0.90)
PLACEMENT_RETRIES = 120
SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 300


def _build_shape_library():
    raw = [
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [1, 0]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1], [0, 1], [0, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 0], [1, 1], [0, 1]],
        [[0, 1], [1, 1], [1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 1], [0, 1], [1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    dedup: list[np.ndarray] = []
    for spec in raw:
        mask = np.array(spec, dtype=bool)
        key = _canonical_key(mask)
        if int(np.sum(mask)) > SHAPE_MAX_AREA:
            continue
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _orientation_key(mask: np.ndarray):
    return mask.shape, mask.tobytes()


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
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= th - 2 and shape.shape[1] <= tw - 2 and int(np.sum(shape)) >= 2
    ]
    if len(available) < MIN_HOLES:
        return None

    k_target = random.randint(HOLE_COUNT_RANGE[0], min(HOLE_COUNT_RANGE[1], len(available), NUM_COLORS - 2))
    random.shuffle(available)

    strong_asym = [shape for shape in available if len(_all_orientations(shape)) >= 4]
    selected: list[np.ndarray] = []
    if strong_asym:
        chosen = random.choice(strong_asym)
        selected.append(chosen)
        available = [shape for shape in available if _canonical_key(shape) != _canonical_key(chosen)]

    for shape in available:
        if len(selected) >= k_target:
            break
        selected.append(shape)

    if len(selected) < MIN_HOLES:
        return None

    random.shuffle(selected)
    selected.sort(key=lambda shape: (int(np.sum(shape)), shape.shape[0] * shape.shape[1]), reverse=True)
    return selected


def _build_hole_layout(th: int, tw: int):
    """
    Place one instance for each chosen canonical shape.
    Returns (occ, hole_records) where each record stores canonical shape, oriented hole mask, and top-left position.
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
                    "hole_pos": (pr, pc),
                }
            )
            placed = True
            break
        if not placed:
            return None

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    return occ, hole_records


def _build_output(occ: np.ndarray, hole_records: list[dict[str, object]], frame_col: int, bg: int):
    """Render the target output and assign one unique color to each hole shape."""
    avail_colors = [color for color in range(NUM_COLORS) if color not in (bg, frame_col)]
    if len(avail_colors) < len(hole_records):
        return None

    output = np.full(occ.shape, frame_col, dtype=int)
    fill_colors = random.sample(avail_colors, len(hole_records))
    shape_info: list[dict[str, object]] = []

    for record, fill_col in zip(hole_records, fill_colors):
        hole_mask = record["hole_mask"]
        pr, pc = record["hole_pos"]
        patch = np.full(hole_mask.shape, fill_col, dtype=int)
        paste(output, patch, int(pr), int(pc), mask=hole_mask)
        shape_info.append(
            {
                "base_shape": record["base_shape"],
                "hole_mask": hole_mask,
                "hole_pos": (int(pr), int(pc)),
                "fill_color": int(fill_col),
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


def _scatter_sprites(inp: np.ndarray, shape_info: list[dict[str, object]], bg: int):
    """
    Place exactly one sprite for each hole shape.
    Returns the number of pairs whose sprite orientation differs from the hole orientation.
    """
    placement = inp.copy()
    mismatch_needed = any(len(_all_orientations(info["base_shape"])) > 1 for info in shape_info)
    mismatch_count = 0

    for info in shape_info:
        base_shape = info["base_shape"]
        hole_mask = info["hole_mask"]
        fill_color = info["fill_color"]
        orientations = _all_orientations(base_shape)
        hole_key = _orientation_key(hole_mask)
        diff_orientations = [ori for ori in orientations if _orientation_key(ori) != hole_key]

        if mismatch_needed and diff_orientations:
            oriented = random.choice(diff_orientations)
        else:
            oriented = random.choice(orientations)

        if not _place_sprite(inp, placement, oriented, int(fill_color), bg):
            return None

        if _orientation_key(oriented) != hole_key:
            mismatch_count += 1
            mismatch_needed = False

    return mismatch_count


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt for the 1:1 shape-to-sprite variant."""
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

    mismatch_count = _scatter_sprites(inp, shape_info, bg)
    if mismatch_count is None:
        return None

    try:
        if np.array_equal(main(inp), output):
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
