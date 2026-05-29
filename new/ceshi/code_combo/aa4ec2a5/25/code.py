# Combo variant of aa4ec2a5 (index 25)
# combo_spec: D2/0+D5/2+D8/0
# Name: 按洞数量分级+按洞形状编码色+洞按面积分色填充

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# ---------------
# a) Concrete fused behavior and trace
#    - Input still shows several blue 4-connected objects on one uniform background, but now the
#      hollow objects may contain 1, 2, or 3 enclosed holes, and all holes inside one hollow
#      object share a visible geometric family: rectangle / L-shape / irregular.
#    - Output still adds the same 1-pixel red 8-neighborhood outline around each object.
#    - The original hollow logic was:
#        detect holes -> recolor whole object to teal -> recolor every hole to magenta.
#      The fused logic is:
#        detect all enclosed holes -> count them into tiers 1 / 2 / 3+ ->
#        classify the common hole shape family of that object ->
#        use (count tier, shape family) together to choose the body color ->
#        recolor each individual hole by its own area band:
#          1-2 -> magenta(6), 3-5 -> yellow(4), >5 -> green(3) ->
#        seal holes into the temporary solid mask -> draw the outer red outline.
#    - Example trace:
#      a 2-hole object with two L-shaped holes used to become one fixed teal ring with magenta
#      holes. In this combo it becomes the body color assigned to (2 holes, L-shape), while one
#      hole may stay yellow if its area is 3-5 and the other may turn green if its area is >5.
#
# b) Complexity calibration
#    - The body color depends on exactly two low-cardinality features:
#      hole-count tier in {1, 2, 3+} and shared hole-shape family in {rectangle, lshape, irregular}.
#      That keeps the rule structured rather than free-form.
#    - The palette budget is finite: after reserving blue input, red outline, and one background,
#      there are only eight practical body colors for nine count×shape cells. The mapping therefore
#      uses one deliberate repeat across two non-competing cells, while keeping every row and every
#      column internally distinct. This preserves both count-dependence and shape-dependence.
#    - The generator makes every sample contain exactly three hollow objects covering the three
#      count tiers and the three shape families, so each training example contains direct contrasts.
#
# c) Implicit assumption audit for main()
#    - `_interior_holes(...)` still assumes holes are 4-connected background regions fully inside
#      the object's bbox. The generator preserves that.
#    - The original `if holes: out[mask] = 8` assumed one universal hollow-object color. That is
#      invalid now and is replaced by `_body_color(len(holes), _shape_class_for_holes(...))`.
#    - A single-hole D8 solver could classify one hole directly. In this combo, multi-hole objects
#      need an object-level shape rule. The chosen rule is: if all holes share the same family, use
#      that family; otherwise treat the object as irregular. The generator always constructs same-
#      family holes per hollow object so the intended family is stable and human-visible.
#    - `solid[hole] = True` remains necessary; without it each interior hole would receive an
#      incorrect red border. This is still true even when a hole is yellow like no longer the bg.
#    - `out[outline & (out == bg)] = 2` still assumes outlines only write onto untouched exterior
#      background. The placement constraints maintain that separation.
#
# d) Placement legality
#    - Hollow objects are generated from a solid rectangle, then several separated hole masks are
#      carved with at least a 1-cell foreground wall to the exterior and at least a 2-cell gap
#      between hole slots. Optional outside extensions only add foreground, so they cannot merge
#      holes or leak them to the outside.
#    - Components are placed with Chebyshev distance >= 3 and border clearance >= 1, so red
#      outlines remain disjoint and stay within the canvas.
#
# e) Visual observability
#    - Each sample shows all three count tiers and all three shape families across its three hollow
#      objects. This gives a solver visible evidence that "body color is not just hole/no-hole".
#    - Hole-area coloring is also made visible in every sample by forcing the global set of holes
#      to cover small, medium, and large area bands.
#    - Yellow holes would be ambiguous on a yellow background, so the background is changed to
#      grey(5). This is a necessary support change for D5 compatibility, not a new hidden rule.
#
# f) Visual cleanliness
#    - Holes inside one object all share one shape family, which makes the object-level shape cue
#      readable rather than noisy.
#    - Multiple holes are arranged in separated slots with simple layouts, avoiding cluttered
#      overlaps or very thin unreadable corridors.
#
# g) Change inferability
#    - Hole colors use only the three D5 size bands, with repeated support in every sample.
#    - Body colors come from a fixed 3x3 lookup over count tier × shape family. The generator
#      randomizes which shape gets which count tier across samples, so the two factors can be
#      disentangled from a small ARC training set instead of collapsing into one object identity.


OUTLINE_COLOR = 2
BG_COLOR = 5
FG_COLOR = 1

COUNT_TIER_COLORS = {
    1: {"rectangle": 8, "lshape": 3, "irregular": 7},
    2: {"rectangle": 4, "lshape": 9, "irregular": 6},
    3: {"rectangle": 0, "lshape": 1, "irregular": 8},
}

GRID_H_RANGE = (24, 34)
GRID_W_RANGE = (24, 34)
COMP_COUNT_RANGE = (4, 5)
HOLLOW_COUNT = 3

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

HOLLOW_GEN_RETRIES = 80
HOLLOW_MIN_AREA = 14
HOLLOW_OUTER_MARGIN_RANGE = (1, 3)
HOLLOW_ROW_GAP_RANGE = (2, 3)
HOLLOW_COL_GAP_RANGE = (2, 3)
HOLLOW_MAX_EXTENSIONS = 2
HOLLOW_EXTENSION_PROB = 0.65
HOLLOW_EXT_H_RANGE = (1, 3)
HOLLOW_EXT_W_RANGE = (1, 3)

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 500
GEN_OUTER_RETRIES = 500

RECT_AREA_OPTIONS = {
    "small": [1, 2],
    "medium": [3, 4, 5],
    "large": [6, 8, 9, 10, 12],
}

IRREGULAR_TEMPLATES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),            # area 4
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),  # area 5
    np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),  # area 6
    np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),  # area 7
    np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),  # area 8
]


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords.copy()
    shifted[:, 0] -= r0
    shifted[:, 1] -= c0
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for hole in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == sub.shape[0] - 1)
            | (hole[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = hole.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            holes.append(shifted)
    return holes


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _is_l_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2 or bool(mask.all()):
        return False

    for arm_h in range(1, h):
        for arm_w in range(1, w):
            missing_h = h - arm_h
            missing_w = w - arm_w
            if missing_h * missing_w < 1:
                continue

            expected = np.zeros((h, w), dtype=bool)
            expected[:arm_h, :] = True
            expected[:, :arm_w] = True
            if np.array_equal(mask, expected):
                return True
            if np.array_equal(mask, np.fliplr(expected)):
                return True
            if np.array_equal(mask, np.flipud(expected)):
                return True
            if np.array_equal(mask, np.flipud(np.fliplr(expected))):
                return True
    return False


def _hole_kind(hole: np.ndarray) -> str:
    tight = _tight_mask(hole)
    if bool(tight.all()):
        return "rectangle"
    if _is_l_hole(tight):
        return "lshape"
    return "irregular"


def _shape_class_for_holes(holes: list[np.ndarray]) -> str | None:
    if not holes:
        return None
    kinds = {_hole_kind(hole) for hole in holes}
    if len(kinds) == 1:
        return next(iter(kinds))
    return "irregular"


def _count_tier(num_holes: int) -> int:
    if num_holes <= 0:
        return 0
    if num_holes == 1:
        return 1
    if num_holes == 2:
        return 2
    return 3


def _body_color(num_holes: int, shape_kind: str) -> int:
    return COUNT_TIER_COLORS[_count_tier(num_holes)][shape_kind]


def _hole_fill_color(area: int) -> int:
    if area <= 2:
        return 6
    if area <= 5:
        return 4
    return 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around objects, while jointly encoding hole count, hole shape, and hole area.

    Input structure:
    - A uniform grey background contains several blue 4-connected foreground components.
    - Some components are solid; others enclose one or more fully surrounded 4-connected
      background holes inside their bounding boxes.

    Rule:
    - Every foreground component gets a one-cell red outline over its 8-neighborhood exterior.
    - Solid components stay blue (1).
    - If a component has holes, first count them into tiers 1 / 2 / 3+, then classify the
      object's hole family as rectangle / L-shape / irregular, and recolor the whole object by
      the fixed lookup over those two features.
    - Recolor each enclosed hole independently by area: 1-2 cells -> magenta (6),
      3-5 cells -> yellow (4), and more than 5 cells -> green (3).

    Key insight: The outer border is still drawn around the object's sealed silhouette, but the
    interior recoloring now splits into two coupled decisions: the body color depends jointly on
    hole count and shared hole geometry, while each hole's own fill depends only on its area.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            shape_kind = _shape_class_for_holes(holes)
            out[mask] = _body_color(len(holes), shape_kind)
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = _hole_fill_color(len(hole))
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


def _rect_dims(area: int, dim_max: int = 5) -> tuple[int, int] | None:
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= dim_max and area // i <= dim_max
    ]
    if not factors:
        return None
    h, w = random.choice(factors)
    if random.random() < 0.5:
        h, w = w, h
    return h, w


def _chebyshev_conflict(
    rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray
) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _make_polyomino() -> np.ndarray | None:
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [
        (random.randint(*POLY_RECT_DIM_RANGE), random.randint(*POLY_RECT_DIM_RANGE))
        for _ in range(num_rects)
    ]

    canvas_h = sum(r[0] for r in rects) + POLY_CANVAS_PAD
    canvas_w = sum(r[1] for r in rects) + POLY_CANVAS_PAD
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r0, c0 = 2, 2
    mask[r0 : r0 + rects[0][0], c0 : c0 + rects[0][1]] = True

    for i in range(1, num_rects):
        rh, rw = rects[i]
        for _ in range(POLY_PLACEMENT_RETRIES):
            nr = random.randint(max(0, r0 - rh + 1), r0 + rects[0][0] + 2)
            nc = random.randint(max(0, c0 - rw + 1), c0 + rects[0][1] + 2)
            if nr + rh > canvas_h or nc + rw > canvas_w or nr < 0 or nc < 0:
                continue
            if mask[nr : nr + rh, nc : nc + rw].any():
                mask[nr : nr + rh, nc : nc + rw] = True
                break

    mask = _crop_mask(mask)
    if mask is None or not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        return np.ones((1, 2), dtype=bool) if random.random() < 0.5 else np.ones((2, 1), dtype=bool)
    if size == 3:
        return random.choice(
            [
                np.array([[1, 1], [1, 0]], dtype=bool),
                np.array([[1, 0], [1, 1]], dtype=bool),
                np.array([[1, 1, 1]], dtype=bool),
                np.array([[1], [1], [1]], dtype=bool),
            ]
        )
    return np.ones((2, 2), dtype=bool)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    out = np.rot90(out, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _make_rect_hole_mask(area: int) -> np.ndarray | None:
    dims = _rect_dims(area, dim_max=5)
    if dims is None:
        return None
    return np.ones(dims, dtype=bool)


def _make_l_hole_mask(area: int) -> np.ndarray | None:
    for h in range(2, 6):
        for w in range(2, 6):
            for arm_h in range(1, h):
                for arm_w in range(1, w):
                    mask = np.zeros((h, w), dtype=bool)
                    mask[:arm_h, :] = True
                    mask[:, :arm_w] = True
                    if int(mask.sum()) == area:
                        return _transform_mask(mask)
    return None


def _make_irregular_hole_mask(area: int) -> np.ndarray | None:
    matches = [tpl for tpl in IRREGULAR_TEMPLATES if int(tpl.sum()) == area]
    if not matches:
        return None
    return _transform_mask(random.choice(matches))


def _make_hole_mask(kind: str, area: int) -> np.ndarray | None:
    if kind == "rectangle":
        return _make_rect_hole_mask(area)
    if kind == "lshape":
        return _make_l_hole_mask(area)
    if kind == "irregular":
        return _make_irregular_hole_mask(area)
    raise ValueError(f"Unknown hole kind: {kind}")


def _choose_area_sequence(kind: str, num_holes: int) -> list[int]:
    if kind == "rectangle":
        base = [2, 4, 8]
        extras = [1, 5, 6, 9]
    elif kind == "lshape":
        base = [3, 6, 5]
        extras = [7, 8]
    else:
        base = [4, 7, 5]
        extras = [4, 6, 7]

    areas = base[:num_holes]
    while len(areas) < num_holes:
        areas.append(random.choice(extras))
    random.shuffle(areas)
    return areas


def _layout_hole_masks(
    hole_masks: list[np.ndarray],
) -> tuple[int, int, list[tuple[int, int, np.ndarray]]]:
    n = len(hole_masks)
    if n == 1:
        rows = [hole_masks]
    elif n == 2:
        rows = [hole_masks] if random.random() < 0.6 else [[hole_masks[0]], [hole_masks[1]]]
    elif n == 3:
        rows = [hole_masks[:2], [hole_masks[2]]] if random.random() < 0.7 else [hole_masks]
    else:
        rows = [hole_masks[:2], hole_masks[2:]]

    placements: list[tuple[int, int, np.ndarray]] = []
    row_top = 0
    layout_w = 0
    for row_idx, row in enumerate(rows):
        row_h = max(mask.shape[0] for mask in row)
        gap = random.randint(*HOLLOW_COL_GAP_RANGE)
        row_w = sum(mask.shape[1] for mask in row) + gap * max(0, len(row) - 1)
        layout_w = max(layout_w, row_w)

        col_left = 0
        for hole_mask in row:
            hh, hw = hole_mask.shape
            offset = random.randint(0, row_h - hh)
            placements.append((row_top + offset, col_left, hole_mask))
            col_left += hw + gap

        if row_idx != len(rows) - 1:
            row_top += row_h + random.randint(*HOLLOW_ROW_GAP_RANGE)
        else:
            row_top += row_h

    return row_top, layout_w, placements


def _add_extension(mask: np.ndarray) -> np.ndarray:
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    base_h, base_w = mask.shape
    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)
    ext_w = random.randint(*HOLLOW_EXT_W_RANGE)

    if side in ("top", "bottom"):
        width = random.randint(max(2, base_w // 2), base_w)
        offset = random.randint(0, base_w - width)
        new_mask = np.zeros((base_h + ext_h, base_w), dtype=bool)
        if side == "top":
            new_mask[ext_h:, :] = mask
            new_mask[:ext_h, offset : offset + width] = True
        else:
            new_mask[:base_h, :] = mask
            new_mask[base_h:, offset : offset + width] = True
        return new_mask

    height = random.randint(max(2, base_h // 2), base_h)
    offset = random.randint(0, base_h - height)
    new_mask = np.zeros((base_h, base_w + ext_w), dtype=bool)
    if side == "left":
        new_mask[:, ext_w:] = mask
        new_mask[offset : offset + height, :ext_w] = True
    else:
        new_mask[:, :base_w] = mask
        new_mask[offset : offset + height, base_w:] = True
    return new_mask


def _hole_areas_in_mask(mask: np.ndarray) -> list[int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    return sorted(len(hole) for hole in _interior_holes(mask, coords))


def _shape_class_in_mask(mask: np.ndarray) -> str | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _shape_class_for_holes(_interior_holes(mask, coords))


def _make_hollow_component(kind: str, num_holes: int) -> np.ndarray | None:
    target_areas = _choose_area_sequence(kind, num_holes)
    hole_masks: list[np.ndarray] = []
    for area in target_areas:
        hole_mask = _make_hole_mask(kind, area)
        if hole_mask is None:
            return None
        hole_masks.append(hole_mask)

    for _ in range(HOLLOW_GEN_RETRIES):
        shuffled = hole_masks[:]
        random.shuffle(shuffled)
        layout_h, layout_w, placements = _layout_hole_masks(shuffled)

        pad_top = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
        pad_bottom = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
        pad_left = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
        pad_right = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)

        mask = np.ones((layout_h + pad_top + pad_bottom, layout_w + pad_left + pad_right), dtype=bool)
        for hole_r, hole_c, hole_mask in placements:
            rr = pad_top + hole_r
            cc = pad_left + hole_c
            sub = mask[rr : rr + hole_mask.shape[0], cc : cc + hole_mask.shape[1]]
            sub[hole_mask] = False
            mask[rr : rr + hole_mask.shape[0], cc : cc + hole_mask.shape[1]] = sub

        for _ in range(random.randint(0, HOLLOW_MAX_EXTENSIONS)):
            mask = _add_extension(mask)

        mask = _crop_mask(mask)
        if mask is None:
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue
        if int(mask.sum()) < HOLLOW_MIN_AREA:
            continue
        if _hole_areas_in_mask(mask) != sorted(target_areas):
            continue
        if len(_interior_holes(mask, np.argwhere(mask))) != num_holes:
            continue
        if _shape_class_in_mask(mask) != kind:
            continue
        return mask
    return None


def _sample_hollow_specs() -> list[tuple[int, str]]:
    counts = [1, 2, 3]
    shapes = ["rectangle", "lshape", "irregular"]
    random.shuffle(shapes)
    return list(zip(counts, shapes))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    num_components = random.randint(*COMP_COUNT_RANGE)

    hollow_specs = _sample_hollow_specs()
    components: list[np.ndarray] = []

    for num_holes, kind in hollow_specs:
        mask = _make_hollow_component(kind, num_holes)
        if mask is None:
            return None
        components.append(mask)

    while len(components) < num_components:
        if random.random() < SMALL_SHAPE_PROB:
            components.append(_make_small_shape())
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append(mask)
                    break
            else:
                return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        if mh > H - 2 * BORDER_MIN or mw > W - 2 * BORDER_MIN:
            return None

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, H - mh - BORDER_MIN)
            c = random.randint(BORDER_MIN, W - mw - BORDER_MIN)

            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, pm in placed:
                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        out = main(grid)
    except Exception:
        return None

    hollow_counts: list[int] = []
    shape_classes: list[str] = []
    area_bands: set[str] = set()
    for comp in find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True):
        mask = _coords_mask(grid.shape, comp["coords"])
        holes = _interior_holes(mask, comp["coords"])
        if not holes:
            continue
        hollow_counts.append(len(holes))
        shape_classes.append(_shape_class_for_holes(holes))
        for hole in holes:
            area = len(hole)
            if area <= 2:
                area_bands.add("small")
            elif area <= 5:
                area_bands.add("medium")
            else:
                area_bands.add("large")

    if sorted(hollow_counts) != [1, 2, 3]:
        return None
    if sorted(shape_classes) != ["irregular", "lshape", "rectangle"]:
        return None
    if area_bands != {"small", "medium", "large"}:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
