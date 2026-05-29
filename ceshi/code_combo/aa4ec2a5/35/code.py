# Combo variant of aa4ec2a5 (index 35)
# combo_spec: D8/1+D11/1
# Name: 按空间分组+着色由位置决定
#
# Design analysis
# a) Concrete behavior and trace:
#    The original task treats each blue component independently: every component gets a red
#    8-neighbor outline, and only hollow components are recolored inside (body teal, hole
#    magenta). D8/1 changes the hollow-body color to a left/right positional choice. D11/1
#    changes the unit of reasoning from individual objects to spatial groups found by the
#    largest horizontal empty corridor. The fused rule keeps both changes by first splitting
#    all components into a left group and a right group, then computing one shared hollow-body
#    color per group from that group's pooled foreground centroid: left-group centroid -> teal,
#    right-group centroid -> green. Example trace: suppose the left cluster contains two hollow
#    objects and the right cluster contains one hollow plus one solid. Under the original task,
#    all three hollow objects would become teal independently. Under pure D8/1, a wide hollow
#    object whose own centroid crosses the global midpoint would become green even if it still
#    belongs to the left cluster. Under this fused variant, both hollow objects in the left
#    cluster become teal together because the group centroid is still on the left, while the
#    right-cluster hollow becomes green. That wide left-cluster hollow is the intentional
#    conflict case showing that grouping overrides per-object positional recoloring.
# b) Complexity calibration:
#    The grouping rule is fixed to exactly one horizontal split and the color choice still has
#    only two buckets (left=teal, right=green). To make the grouped behavior visible, the
#    generator always builds two clusters of two objects each, with two hollow objects in the
#    left cluster and one hollow object in the right cluster. This creates repeated evidence
#    that same-group hollow objects share one scheme instead of being colored independently.
# c) Hidden-assumption audit for main():
#    - find_components(... colors=[1]) still assumes the input foreground is monochrome blue.
#      The generator preserves that, so component detection remains valid.
#    - _interior_holes() still relies only on geometry inside each component bbox. Grouping does
#      not alter local hole detection, so this remains safe.
#    - The original one-pass loop could color each hollow component immediately. That is no
#      longer valid because a hollow component's color now depends on the pooled centroid of its
#      spatial group, so main() must collect and split records first, then render second.
#    - The original fixed "out[mask] = 8" is invalid because hollow color is no longer global.
#      It must be replaced by a per-group color decision.
#    - out[outline & (out == bg)] = 2 remains safe because outlines are still drawn only onto
#      background cells; placement guarantees that outlines do not need to overwrite bodies/holes.
# d) Placement legality:
#    The generator preserves border clearance and pairwise Chebyshev spacing. In addition, it
#    enforces a dominant inter-group horizontal gap so the solver's "largest gap" split recovers
#    the intended 2+2 grouping. The wide conflict hollow is only accepted when it fits entirely
#    inside the left cluster without breaking that split.
# e) Visual observability:
#    A human can see two horizontal clusters separated by a clear blank corridor. Inside the left
#    cluster, both hollow objects share teal even when one stretches far enough right that D8/1
#    alone would have colored it green. The right cluster supplies repeated green evidence. This
#    makes the fusion observable from a few examples instead of hiding it in a single edge case.
# f) Visual cleanliness:
#    The output still uses only five semantic colors: blue solid bodies, teal/green hollow bodies,
#    magenta holes, red outlines, yellow background. Clusters are spaced apart and outlines remain
#    disjoint, so the grouped recoloring is easy to read.
# g) Change inferability:
#    The only new abstraction is "group first, then apply one left/right scheme to that whole
#    group." There are no invisible ids or arbitrary orderings. The left/right decision is still
#    spatial, but now it is tied to the visible cluster rather than to each hollow object alone.
# h) Anti-degeneration checks:
#    A sample would degenerate back to pure D8/1 if every hollow object happened to receive the
#    same color from both rules. generate_input() therefore rejects any sample whose grouped color
#    assignment matches the per-object D8/1 color on all hollow objects.
# i) New solver assumptions and generator guarantees:
#    main() assumes there are exactly two visible spatial groups separated by the widest horizontal
#    corridor, and that group pooled centroids land on opposite halves of the grid. generate_input()
#    guarantees this and additionally ensures at least one hollow object disagrees with its own
#    individual left/right color, so the fusion rule is actually exercised.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


LEFT_HOLLOW_COLOR = 8
RIGHT_HOLLOW_COLOR = 3
HOLE_COLOR = 6
OUTLINE_COLOR = 2
SOLID_COLOR = 1


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
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[SOLID_COLOR]):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        records.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "mask": mask,
                "holes": holes,
                "solid": solid,
            }
        )
    records.sort(key=lambda rec: (int(rec["bbox"][1]), int(rec["bbox"][0])))
    return records


def _individual_hollow_color(coords: np.ndarray, grid_w: int) -> int:
    return LEFT_HOLLOW_COLOR if float(coords[:, 1].mean()) < (grid_w / 2.0) else RIGHT_HOLLOW_COLOR


def _group_hollow_color(group_records: list[dict], grid_w: int) -> int:
    pooled_cols = np.concatenate([np.asarray(rec["coords"])[:, 1] for rec in group_records])
    return LEFT_HOLLOW_COLOR if float(pooled_cols.mean()) < (grid_w / 2.0) else RIGHT_HOLLOW_COLOR


def _group_metrics(records: list[dict], grid_w: int) -> dict:
    split = _group_split_index(records)
    left_records = records[:split]
    right_records = records[split:]
    groups = [left_records, right_records]
    group_colors = [_group_hollow_color(group, grid_w) for group in groups if group]

    conflict_count = 0
    hollow_count = 0
    for group in groups:
        if not group:
            continue
        group_color = _group_hollow_color(group, grid_w)
        for rec in group:
            if not rec["holes"]:
                continue
            hollow_count += 1
            if _individual_hollow_color(np.asarray(rec["coords"]), grid_w) != group_color:
                conflict_count += 1

    return {
        "split": split,
        "groups": groups,
        "group_colors": group_colors,
        "conflict_count": conflict_count,
        "hollow_count": hollow_count,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around objects, with hollow-body colors chosen per spatial group.

    Input structure:
    - A uniform yellow background contains several blue 4-connected components.
    - The components form two horizontal clusters separated by the widest empty vertical corridor.
    - Some components contain enclosed 4-connected background holes fully inside their bounding boxes.

    Rule:
    - First split all blue components into a left group and a right group at the largest horizontal gap.
    - For each group, pool all foreground cells in that group and inspect the pooled centroid column.
    - If that group centroid lies in the left half, recolor every hollow object in the group teal (8);
    - if it lies in the right half, recolor every hollow object in the group green (3).
    - Recolor every enclosed hole magenta (6), keep solid objects blue (1), then draw a one-cell
      red 8-neighborhood outline (2) around each component using the object-plus-hole mask.

    Key insight:
    - Local hole detection is still per object, but the hollow-body color is no longer decided
      object-by-object. You must first discover the two spatial groups, then apply one left/right
      color choice to every hollow object in the same group.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    records = _component_records(g, bg)
    metrics = _group_metrics(records, g.shape[1])

    for group in metrics["groups"]:
        if not group:
            continue
        group_color = _group_hollow_color(group, g.shape[1])
        for rec in group:
            mask = np.asarray(rec["mask"], dtype=bool)
            holes = rec["holes"]
            out[mask] = SOLID_COLOR if not holes else group_color
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = HOLE_COLOR

            outline = _outline_mask(np.asarray(rec["solid"], dtype=bool))
            out[outline & (out == bg)] = OUTLINE_COLOR

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (30, 30)
BG_COLOR = 4
FG_COLOR = SOLID_COLOR

GROUP_COUNT = 2
COMPONENTS_PER_GROUP = 2
TOTAL_COMPONENTS = GROUP_COUNT * COMPONENTS_PER_GROUP
LEFT_GROUP_ID = 0
RIGHT_GROUP_ID = 1

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 40

SMALL_SHAPE_PROB = 0.45
SMALL_SHAPE_SIZE_RANGE = (2, 4)

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 80
HOLLOW_MIN_AREA = 10

MAX_COMPONENT_H = 10
MAX_COMPONENT_W = 13
LEFT_SUPPORT_MAX_W = 5
RIGHT_HOLLOW_MAX_W = 5
RIGHT_SOLID_MAX_W = 4
CONFLICT_HOLLOW_MIN_W = 9
CONFLICT_HOLLOW_MIN_LOCAL_CENTROID = 5.0

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 400
PLACEMENT_OPTION_RETRIES = 30
GROUP_GAP_MIN = 4
LEFT_CONFLICT_OFFSET_RANGE = (2, 6)
RIGHT_SOLID_OFFSET_RANGE = (0, 1)
CONFLICT_MARGIN = 0.0

GEN_OUTER_RETRIES = 600


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


def _fits_component(mask: np.ndarray | None, *, max_w: int = MAX_COMPONENT_W) -> bool:
    return mask is not None and mask.shape[0] <= MAX_COMPONENT_H and mask.shape[1] <= max_w


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
    return mask if _fits_component(mask) else None


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
        if cropped is not None and _has_interior_hole(cropped) and _fits_component(cropped):
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


def _make_solid_component(max_w: int = RIGHT_SOLID_MAX_W) -> np.ndarray | None:
    if random.random() < SMALL_SHAPE_PROB:
        mask = _make_small_shape()
        return mask if _fits_component(mask, max_w=max_w) else None
    for _ in range(POLY_GEN_RETRIES):
        mask = _make_polyomino()
        if mask is not None and _fits_component(mask, max_w=max_w):
            return mask
    return None


def _make_filtered_hollow(
    *,
    min_w: int = 0,
    max_w: int = MAX_COMPONENT_W,
    min_local_centroid: float | None = None,
) -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
        if mask is None or mask.sum() < HOLLOW_MIN_AREA or not _fits_component(mask, max_w=max_w):
            continue
        if mask.shape[1] < min_w:
            continue
        if min_local_centroid is not None and _local_centroid_col(mask) < min_local_centroid:
            continue
        return mask
    return None


def _local_centroid_col(mask: np.ndarray) -> float:
    return float(np.where(mask)[1].mean())


def _group_centroid_from_specs(specs: list[dict[str, object]]) -> float:
    total_weight = 0
    total_col = 0.0
    for spec in specs:
        mask = np.asarray(spec["mask"], dtype=bool)
        col = int(spec["col"])
        weight = int(mask.sum())
        total_weight += weight
        total_col += weight * (col + _local_centroid_col(mask))
    return total_col / float(total_weight)


def _placement_options(component_specs: dict[str, np.ndarray], grid_w: int) -> list[dict[str, int]]:
    left_support = component_specs["left_support"]
    left_conflict = component_specs["left_conflict"]
    right_hollow = component_specs["right_hollow"]
    right_solid = component_specs["right_solid"]

    mid = grid_w / 2.0
    options: list[dict[str, int]] = []

    for left_conflict_offset in range(LEFT_CONFLICT_OFFSET_RANGE[0], LEFT_CONFLICT_OFFSET_RANGE[1] + 1):
        left_extent = max(left_support.shape[1], left_conflict_offset + left_conflict.shape[1])
        if left_extent > MAX_COMPONENT_W + LEFT_CONFLICT_OFFSET_RANGE[1]:
            continue
        for right_solid_offset in range(RIGHT_SOLID_OFFSET_RANGE[0], RIGHT_SOLID_OFFSET_RANGE[1] + 1):
            right_extent = max(right_hollow.shape[1], right_solid_offset + right_solid.shape[1])

            left_anchor_max = grid_w - BORDER_MIN - right_extent - GROUP_GAP_MIN - left_extent
            if left_anchor_max < BORDER_MIN:
                continue

            for left_anchor in range(BORDER_MIN, left_anchor_max + 1):
                conflict_centroid = left_anchor + left_conflict_offset + _local_centroid_col(left_conflict)
                if conflict_centroid <= mid + CONFLICT_MARGIN:
                    continue

                left_group_specs = [
                    {"mask": left_support, "col": left_anchor},
                    {"mask": left_conflict, "col": left_anchor + left_conflict_offset},
                ]
                left_group_centroid = _group_centroid_from_specs(left_group_specs)
                if left_group_centroid >= mid:
                    continue

                left_group_right = left_anchor + left_extent - 1
                right_anchor_min = left_group_right + GROUP_GAP_MIN + 1
                right_anchor_max = grid_w - BORDER_MIN - right_extent
                if right_anchor_min > right_anchor_max:
                    continue

                for right_anchor in range(right_anchor_min, right_anchor_max + 1):
                    right_group_specs = [
                        {"mask": right_hollow, "col": right_anchor},
                        {"mask": right_solid, "col": right_anchor + right_solid_offset},
                    ]
                    if _group_centroid_from_specs(right_group_specs) < mid:
                        continue
                    options.append(
                        {
                            "left_support_col": left_anchor,
                            "left_conflict_col": left_anchor + left_conflict_offset,
                            "right_hollow_col": right_anchor,
                            "right_solid_col": right_anchor + right_solid_offset,
                        }
                    )
    return options


def _place_component_at_col(
    H: int,
    W: int,
    mask: np.ndarray,
    col: int,
    placed: list[tuple[int, int, np.ndarray, int, str]],
    group_id: int,
    role: str,
) -> bool:
    mh, mw = mask.shape
    if col < BORDER_MIN or col + mw > W - BORDER_MIN:
        return False

    max_r = H - mh - BORDER_MIN
    if max_r < BORDER_MIN:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(BORDER_MIN, max_r)
        rows_a, cols_a = np.where(mask)
        rows_a, cols_a = rows_a + r, cols_a + col

        conflict = False
        for pr, pc, pm, _group_id, _role in placed:
            row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
            col_gap = max(0, max(col - (pc + pm.shape[1] - 1), pc - (col + mw - 1)))
            if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                continue
            if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
                conflict = True
                break

        if not conflict:
            placed.append((r, col, mask, group_id, role))
            return True
    return False


def _intended_groups_match(grid: np.ndarray, placed: list[tuple[int, int, np.ndarray, int, str]]) -> bool:
    records = _component_records(grid, BG_COLOR)
    if len(records) != TOTAL_COMPONENTS:
        return False

    metrics = _group_metrics(records, grid.shape[1])
    if int(metrics["split"]) != COMPONENTS_PER_GROUP:
        return False
    if metrics["group_colors"] != [LEFT_HOLLOW_COLOR, RIGHT_HOLLOW_COLOR]:
        return False

    membership = []
    for rec in records:
        bbox = rec["bbox"]
        matched = None
        for r, c, mask, group_id, _role in placed:
            mh, mw = mask.shape
            placed_bbox = (r, c, r + mh - 1, c + mw - 1)
            if placed_bbox == bbox:
                matched = group_id
                break
        membership.append(matched)

    return membership == [LEFT_GROUP_ID, LEFT_GROUP_ID, RIGHT_GROUP_ID, RIGHT_GROUP_ID]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    component_specs = {
        "left_support": _make_filtered_hollow(max_w=LEFT_SUPPORT_MAX_W),
        "left_conflict": _make_filtered_hollow(
            min_w=CONFLICT_HOLLOW_MIN_W,
            min_local_centroid=CONFLICT_HOLLOW_MIN_LOCAL_CENTROID,
        ),
        "right_hollow": _make_filtered_hollow(max_w=RIGHT_HOLLOW_MAX_W),
        "right_solid": _make_solid_component(max_w=RIGHT_SOLID_MAX_W),
    }
    if any(mask is None for mask in component_specs.values()):
        return None

    placement_options = _placement_options(
        {key: np.asarray(mask, dtype=bool) for key, mask in component_specs.items()},
        W,
    )
    if not placement_options:
        return None

    role_order = sorted(
        [
            ("left_conflict", LEFT_GROUP_ID),
            ("left_support", LEFT_GROUP_ID),
            ("right_hollow", RIGHT_GROUP_ID),
            ("right_solid", RIGHT_GROUP_ID),
        ],
        key=lambda item: -int(np.asarray(component_specs[item[0]], dtype=bool).sum()),
    )

    option_order = random.sample(placement_options, k=min(len(placement_options), PLACEMENT_OPTION_RETRIES))
    placed: list[tuple[int, int, np.ndarray, int, str]] | None = None
    for placement in option_order:
        trial_placed: list[tuple[int, int, np.ndarray, int, str]] = []
        for role, group_id in role_order:
            mask = np.asarray(component_specs[role], dtype=bool)
            col = int(placement[f"{role}_col"])
            if not _place_component_at_col(H, W, mask, col, trial_placed, group_id, role):
                trial_placed = []
                break
        if trial_placed:
            placed = trial_placed
            break
    if placed is None:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    for r, c, mask, _group_id, _role in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    if not _intended_groups_match(grid, placed):
        return None

    records = _component_records(grid, BG_COLOR)
    metrics = _group_metrics(records, W)
    if metrics["conflict_count"] < 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out == LEFT_HOLLOW_COLOR) or not np.any(out == RIGHT_HOLLOW_COLOR):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
