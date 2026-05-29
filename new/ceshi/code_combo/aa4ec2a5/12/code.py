# Combo variant of aa4ec2a5 (index 12)
# combo_spec: D4/1+D11/1
# Name: 对象按空间分组+轮廓颜色按面积排名

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous
except ModuleNotFoundError:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "solvers"
        if (candidate / "arc_utils.py").exists():
            sys.path.append(str(candidate))
            break
    from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The input still shows only blue foreground objects on a uniform background, but the output now
#    combines two different dependencies. First, objects are partitioned into a left spatial group
#    and a right spatial group using the widest horizontal empty corridor between component bboxes.
#    Group membership controls interior recoloring: the left group uses a cool scheme
#    (solid=blue, hollow shell=teal, hole=magenta), while the right group uses a muted warm scheme
#    (solid=gray, hollow shell=brown, hole=magenta). Second, outline colors ignore groups and are
#    assigned by global area rank across all four objects: largest=red, second=orange,
#    third=yellow, fourth=green. Example: if left solid area 20, left hollow 34, right solid 29,
#    right hollow 14, then the split is left|right by space, the interiors become
#    blue/teal/gray/brown by group+hollowness, and the outlines become orange/red/yellow/green by
#    the global order 34>29>20>14. Relative to the original task, outline color is no longer a
#    constant and object recoloring is no longer independent per component.
# b) Complexity calibration:
#    The generator always creates exactly two spatial groups with two objects each, and each group
#    contains one solid plus one hollow object. This repeats the group-dependent interior scheme
#    twice per sample. Fixing the total to four objects also guarantees four visible outline ranks
#    without introducing too many colors or too many clustering possibilities.
# c) Hidden-assumption audit:
#    The original main() relied on three assumptions that break here:
#    - A component could be rendered as soon as its holes were found. That no longer holds because
#      the interior palette depends on the later spatial split.
#    - A single constant outline color could be written inside the same loop. That no longer holds
#      because outline color now depends on a global area ordering over all components.
#    - Yellow could safely remain the background. That no longer holds because yellow is now the
#      third-rank outline color, so the background is moved to black to keep every rank visible.
#    The solver therefore collects all component records first, assigns group schemes second, and
#    draws ranked outlines only after every area is known.
# d) Placement legality:
#    Components are generated as 4-connected masks and placed with Chebyshev distance >= 3, so
#    one-pixel 8-neighborhood outlines remain disjoint. Each group is anchored in a narrow column
#    band, and the two bands are separated by a guaranteed large horizontal gap, ensuring the
#    solver's largest-gap split matches the intended grouping.
# e) Visual observability:
#    The human-visible cues are explicit: a blank vertical corridor reveals the two groups, both
#    groups repeat their local solid/hollow recolor scheme, and all four outline colors appear in
#    the same sample. No hidden ids, color ordering, or off-grid metadata are used for ranking or
#    grouping. Ranking is based on visible object area only.
# f) Visual cleanliness:
#    Fill colors and outline colors are chosen to avoid collisions: background=black, fills use
#    blue/teal/gray/brown/magenta, and outlines use red/orange/yellow/green. This prevents a
#    ranked outline from being confused with a group-controlled interior color.
# g) Change inferability:
#    The generator rejects tied areas, requires a noticeable area span, requires the top two ranks
#    to be split across the two spatial groups, and requires opposite solid-vs-hollow area order in
#    the two groups. Those checks stop degenerate samples where a solver could fake the rule using
#    only side, hollowness, or a single local comparison instead of the intended fusion logic.
# h) Quantitative anti-degeneration target:
#    On valid generated samples, all four outline rank colors should appear and the left/right group
#    split recovered by the solver should be 2+2. The generator also ensures the combo output is not
#    equal to the original red-outline baseline on the same input.
# i) New solver assumptions and generator guarantees:
#    main() assumes exactly one dominant horizontal gap separates the intended groups, all
#    foreground cells are blue in the input, component areas are distinct, and no outlines overlap.
#    generate_input() enforces each of those conditions directly before accepting a sample.


OUTLINE_RANK_COLORS = [2, 7, 4, 3]
LEFT_SCHEME = {"solid": 1, "hollow": 8, "hole": 6}
RIGHT_SCHEME = {"solid": 5, "hollow": 9, "hole": 6}


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
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


def _outline_color_for_rank(rank: int) -> int:
    return int(OUTLINE_RANK_COLORS[min(rank, len(OUTLINE_RANK_COLORS) - 1)])


def _group_split_index(records: list[dict]) -> int:
    if len(records) <= 1:
        return len(records)
    best_idx = 1
    best_gap = None
    for idx in range(len(records) - 1):
        left_bbox = records[idx]["bbox"]
        right_bbox = records[idx + 1]["bbox"]
        gap = int(right_bbox[1]) - int(left_bbox[3]) - 1
        if best_gap is None or gap > best_gap:
            best_gap = gap
            best_idx = idx + 1
    return best_idx


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    records: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[1]):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        r0, c0, _r1, _c1 = bbox_coords(coords)
        records.append(
            {
                "bbox": bbox_coords(coords),
                "mask": mask,
                "holes": holes,
                "solid": solid,
                "area": int(mask.sum()),
                "anchor": (int(r0), int(c0)),
            }
        )
    records.sort(key=lambda rec: (int(rec["bbox"][1]), int(rec["bbox"][0])))
    return records


def _baseline_original_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[1]):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True
        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor objects by spatial group, then color their outlines by global area rank.

    Input structure:
    - A uniform black background contains four blue 4-connected objects.
    - The objects form two horizontal clusters separated by the widest empty vertical corridor.
    - Each cluster contains one solid object and one hollow object whose enclosed hole is a
      4-connected background region inside the object's bounding box.

    Rule:
    - Split the objects into a left group and a right group at the largest horizontal gap.
    - Apply a group-specific interior palette: left solids stay blue (1), left hollow shells
      become teal (8); right solids become gray (5), right hollow shells become brown (9);
      all enclosed holes become magenta (6).
    - Rank all four objects globally by input area and draw a one-cell 8-neighborhood outline on
      background cells only: largest red (2), second orange (7), third yellow (4), fourth green (3).

    Key insight:
    - The fill color and the outline color come from different reasoning axes. Group membership is
      spatial and local to the left/right clusters, while outline color is a global comparison over
      all object areas. The solver must discover the groups and the rank order before rendering.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    records = _component_records(g, bg)
    split = _group_split_index(records)
    for rec in records[:split]:
        rec["scheme"] = LEFT_SCHEME
    for rec in records[split:]:
        rec["scheme"] = RIGHT_SCHEME

    for rec in records:
        scheme = rec["scheme"]
        mask = rec["mask"]
        holes = rec["holes"]
        out[mask] = int(scheme["hollow"] if holes else scheme["solid"])
        for hole in holes:
            out[hole[:, 0], hole[:, 1]] = int(scheme["hole"])

    ranked = sorted(records, key=lambda rec: (-int(rec["area"]), rec["anchor"][0], rec["anchor"][1]))
    for rank, rec in enumerate(ranked):
        outline = _outline_mask(rec["solid"])
        out[outline & (out == bg)] = _outline_color_for_rank(rank)

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (22, 28)
BG_COLOR = 0
FG_COLOR = 1

GROUP_COUNT = 2
COMPONENTS_PER_GROUP = 2
TOTAL_COMPONENTS = GROUP_COUNT * COMPONENTS_PER_GROUP

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 40

SMALL_SHAPE_PROB = 0.15
SMALL_SHAPE_SIZE_RANGE = (2, 4)

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 60
HOLLOW_MIN_AREA = 10

MAX_COMPONENT_H = 9
MAX_COMPONENT_W = 9

AREA_GAP_MIN = 2
AREA_SPAN_MIN = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300
GROUP_GAP_MIN = 7

GEN_OUTER_RETRIES = 500


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_interior_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for bg_reg in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (bg_reg[:, 0] == 0)
            | (bg_reg[:, 1] == 0)
            | (bg_reg[:, 0] == h - 1)
            | (bg_reg[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            return True
    return False


def _rect_dims(area: int) -> tuple[int, int] | None:
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= HOLE_DIM_MAX and area // i <= HOLE_DIM_MAX
    ]
    if not factors:
        return None
    h, w = random.choice(factors)
    if random.random() < 0.5:
        h, w = w, h
    return h, w


def _chebyshev_conflict(rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _fits_cluster(mask: np.ndarray | None) -> bool:
    return mask is not None and mask.shape[0] <= MAX_COMPONENT_H and mask.shape[1] <= MAX_COMPONENT_W


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
    return mask if _fits_cluster(mask) else None


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int) -> np.ndarray:
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)

    if side in ("top", "bottom"):
        ext_w = min(random.randint(2, max(2, base_w)), base_w)
        offset = random.randint(0, base_w - ext_w)
        new_mask = np.zeros((base_h + ext_h, base_w), dtype=bool)
        if side == "top":
            new_mask[ext_h:, :] = mask
            new_mask[0:ext_h, offset : offset + ext_w] = True
        else:
            new_mask[:base_h, :] = mask
            new_mask[base_h : base_h + ext_h, offset : offset + ext_w] = True
    else:
        ext_h_actual = min(ext_h, base_h)
        ext_w_actual = random.randint(*HOLLOW_EXT_SIDE_W_RANGE)
        offset = random.randint(0, base_h - ext_h_actual)
        new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
        if side == "left":
            new_mask[:, ext_w_actual:] = mask
            new_mask[offset : offset + ext_h_actual, 0:ext_w_actual] = True
        else:
            new_mask[:, :base_w] = mask
            new_mask[offset : offset + ext_h_actual, base_w : base_w + ext_w_actual] = True

    return new_mask


def _carve_hole(mask: np.ndarray, hh: int, hw: int) -> np.ndarray | None:
    mh, mw = mask.shape
    for _ in range(HOLLOW_HOLE_SEARCH_RETRIES):
        hr = random.randint(1, mh - hh - 1)
        hc = random.randint(1, mw - hw - 1)
        if not mask[hr : hr + hh, hc : hc + hw].all():
            continue
        if (
            not mask[hr - 1, hc : hc + hw].any()
            or not mask[hr + hh, hc : hc + hw].any()
            or not mask[hr : hr + hh, hc - 1].any()
            or not mask[hr : hr + hh, hc + hw].any()
        ):
            continue

        carved = mask.copy()
        carved[hr : hr + hh, hc : hc + hw] = False
        if not is_contiguous(carved.astype(int), background=0, connectivity=4):
            continue
        cropped = _crop_mask(carved)
        if cropped is not None and _has_interior_hole(cropped) and _fits_cluster(cropped):
            return cropped
    return None


def _make_hollow_component(target_hole_area: int) -> np.ndarray | None:
    dims = _rect_dims(target_hole_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bot = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, HOLLOW_MAX_NOTCHES)
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        nh = random.randint(1, max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1))
        nw = random.randint(1, max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1))
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, 0:nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    mask = _extend_mask(mask, base_h, base_w)
    return _carve_hole(mask, hh, hw)


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
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


def _make_solid_component() -> np.ndarray | None:
    if random.random() < SMALL_SHAPE_PROB:
        mask = _make_small_shape()
        return mask if _fits_cluster(mask) else None
    for _ in range(POLY_GEN_RETRIES):
        mask = _make_polyomino()
        if mask is not None:
            return mask
    return None


def _place_component(
    H: int,
    W: int,
    mask: np.ndarray,
    col: int,
    placed: list[tuple[int, int, np.ndarray, int, bool]],
    group_id: int,
    hollow: bool,
) -> bool:
    mh, mw = mask.shape
    if col < BORDER_MIN or col + mw > W - BORDER_MIN:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(BORDER_MIN, H - mh - BORDER_MIN)
        rows_a, cols_a = np.where(mask)
        rows_a, cols_a = rows_a + r, cols_a + col

        conflict = False
        for pr, pc, pm, _group_id, _is_hollow in placed:
            row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
            col_gap = max(0, max(col - (pc + pm.shape[1] - 1), pc - (col + mw - 1)))
            if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                continue
            if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
                conflict = True
                break

        if not conflict:
            placed.append((r, col, mask, group_id, hollow))
            return True
    return False


def _ranking_supports_combo(components: list[list[dict]]) -> bool:
    flat = [item for group in components for item in group]
    areas = [int(item["mask"].sum()) for item in flat]
    if len(areas) != TOTAL_COMPONENTS or len(set(areas)) != len(areas):
        return False

    sorted_areas = sorted(areas, reverse=True)
    if sorted_areas[0] - sorted_areas[-1] < AREA_SPAN_MIN:
        return False
    if any(a - b < AREA_GAP_MIN for a, b in zip(sorted_areas, sorted_areas[1:])):
        return False

    hollow_areas = [int(item["mask"].sum()) for item in flat if item["hollow"]]
    solid_areas = [int(item["mask"].sum()) for item in flat if not item["hollow"]]
    if min(hollow_areas) > max(solid_areas) or max(hollow_areas) < min(solid_areas):
        return False

    area_order = sorted(flat, key=lambda item: -int(item["mask"].sum()))
    top_two_groups = {int(item["group"]) for item in area_order[:2]}
    if len(top_two_groups) != 2:
        return False

    cmp0 = int(components[0][0]["mask"].sum()) - int(components[0][1]["mask"].sum())
    cmp1 = int(components[1][0]["mask"].sum()) - int(components[1][1]["mask"].sum())
    if cmp0 == 0 or cmp1 == 0 or (cmp0 > 0) == (cmp1 > 0):
        return False

    return True


def _intended_groups_match(grid: np.ndarray, placed: list[tuple[int, int, np.ndarray, int, bool]]) -> bool:
    records = _component_records(grid, BG_COLOR)
    if len(records) != TOTAL_COMPONENTS:
        return False
    split = _group_split_index(records)
    if split != COMPONENTS_PER_GROUP:
        return False

    membership: list[int | None] = []
    for rec in records:
        bbox = rec["bbox"]
        matched = None
        for r, c, mask, group_id, _hollow in placed:
            mh, mw = mask.shape
            placed_bbox = (r, c, r + mh - 1, c + mw - 1)
            if placed_bbox == bbox:
                matched = group_id
                break
        membership.append(matched)

    return membership == [0, 0, 1, 1]


def _combo_behavior_present(grid: np.ndarray) -> bool:
    out = main(grid)
    if np.array_equal(out, _baseline_original_output(grid)):
        return False

    rank_colors = set(OUTLINE_RANK_COLORS)
    visible_rank_colors = {int(color) for color in np.unique(out) if int(color) in rank_colors}
    if visible_rank_colors != rank_colors:
        return False

    unique = set(int(v) for v in np.unique(out))
    required_fills = {LEFT_SCHEME["solid"], LEFT_SCHEME["hollow"], RIGHT_SCHEME["solid"], RIGHT_SCHEME["hollow"], 6}
    return required_fills.issubset(unique)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    group_components: list[list[dict]] = []
    for group_id in range(GROUP_COUNT):
        hollow_mask = None
        for _ in range(HOLLOW_GEN_RETRIES):
            hollow_mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
            if hollow_mask is not None and int(hollow_mask.sum()) >= HOLLOW_MIN_AREA:
                break
        if hollow_mask is None or int(hollow_mask.sum()) < HOLLOW_MIN_AREA:
            return None

        solid_mask = _make_solid_component()
        if solid_mask is None:
            return None

        group_components.append(
            [
                {"mask": solid_mask, "group": group_id, "hollow": False},
                {"mask": hollow_mask, "group": group_id, "hollow": True},
            ]
        )

    if not _ranking_supports_combo(group_components):
        return None

    left_max_w = max(item["mask"].shape[1] for item in group_components[0])
    right_max_w = max(item["mask"].shape[1] for item in group_components[1])
    if left_max_w + right_max_w + GROUP_GAP_MIN + 2 * BORDER_MIN > W:
        return None

    left_anchor_max = W - BORDER_MIN - right_max_w - GROUP_GAP_MIN - left_max_w
    if left_anchor_max < BORDER_MIN:
        return None
    left_anchor = random.randint(BORDER_MIN, left_anchor_max)
    right_anchor_min = left_anchor + left_max_w + GROUP_GAP_MIN
    right_anchor_max = W - BORDER_MIN - right_max_w
    if right_anchor_min > right_anchor_max:
        return None
    right_anchor = random.randint(right_anchor_min, right_anchor_max)

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray, int, bool]] = []

    for group_id, (anchor, items) in enumerate(zip([left_anchor, right_anchor], group_components)):
        for item in sorted(items, key=lambda rec: -rec["mask"].shape[0]):
            if not _place_component(H, W, item["mask"], anchor, placed, group_id, bool(item["hollow"])):
                return None

    for r, c, mask, _group_id, _hollow in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    if not _intended_groups_match(grid, placed):
        return None
    if not _combo_behavior_present(grid):
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
