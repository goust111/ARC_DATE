# Variant of aa4ec2a5
# Dimension: D11 — 对象间的空间关系与分组 (meta)
# Change: 各蓝色对象独立处理，对象间无特殊空间关系约束... → 对象按空间分组分别处理

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for this D11 variant
# a) Concrete behavior:
#    Input still contains only blue foreground objects on a yellow background. The visible change
#    happens in the output stage: objects are no longer colored independently. Instead, all objects
#    are first split into a left spatial group and a right spatial group by the largest horizontal
#    empty gap between component bboxes. The left group uses the original cool palette
#    (solid=blue, hollow body=teal, holes=magenta), while the right group uses a warm palette
#    (solid=green, hollow body=orange, holes=magenta). Red outlines remain the same.
#    Example trace:
#    - Original: blue solid at left, blue hollow at right -> left stays blue+red outline, right
#      becomes teal/magenta+red outline, with no interaction between them.
#    - Variant: if two left-side objects lie before the widest x-gap, both are interpreted as one
#      spatial group and both must use the left palette; objects after that gap must use the right
#      palette, regardless of whether each individual object is solid or hollow.
# b) Complexity calibration:
#    The grouping rule is fixed to exactly one horizontal split, not free-form clustering. This
#    keeps the inference burden moderate while still forcing a new "find the groups first" step.
#    Generator always creates two visible clusters and two components per cluster so the group rule
#    is repeated within each sample rather than appearing as a one-off accident.
# c) Hidden assumptions audited against the original main():
#    - The original loop assumed each component could be colored immediately after local hole
#      detection. That is no longer valid because the color now depends on group membership.
#    - The original unconditional "out[mask] = 8" was harmless because all hollow objects shared
#      one recolor. In the variant, recolor must be selected after grouping, so component metadata
#      must be collected first and rendered second.
#    - "out[outline & (out == bg)] = 2" remains valid because outlines are still red and placement
#      ensures they never need to overwrite another object/group color.
# d) Placement legality:
#    Generator uses four components total: one hollow + one solid in the left cluster and one
#    hollow + one solid in the right cluster. The two clusters are separated by a guaranteed wide
#    horizontal gap, and the original Chebyshev spacing is preserved so outlines do not collide.
# e) Visual observability:
#    The split is visible from the blank vertical corridor between clusters, not from hidden ids.
#    Left/right palette mapping is fixed across samples; it does not randomize per instance.
# f) Visual cleanliness:
#    Cluster-local columns are narrow and outlines stay disjoint, so there is no dense overlap of
#    colored marks. Magenta only appears inside hollow objects, which keeps the output legible.
# g) Color inference:
#    Only two palettes are used and both are tied to an obvious spatial cue (left group vs right
#    group). This prevents arbitrary color memorization.
# h) Generator anti-degeneration check:
#    Each generated sample is verified so that the solver's own horizontal split recovers the
#    intended 2+2 grouping. If not, the layout is rejected and regenerated.
# i) New solver assumptions and generator guarantees:
#    main() assumes the largest horizontal gap separates the intended groups and that all foreground
#    objects are blue. generate_input() guarantees exactly that by producing two blue clusters with
#    a dominant inter-cluster gap.


LEFT_SCHEME = {"solid": 1, "hollow": 8, "hole": 6}
RIGHT_SCHEME = {"solid": 3, "hollow": 7, "hole": 6}


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
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[1]):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around grouped objects, using a group-dependent recolor palette for each side.

    Input structure:
    - A uniform yellow background contains several blue 4-connected components.
    - The components form two horizontal clusters separated by the widest empty vertical corridor.
    - Some components may contain enclosed 4-connected background holes inside their bbox.

    Rule:
    - First split the blue components into a left group and a right group at the largest horizontal gap.
    - For the left group, solid objects stay blue (1) and hollow objects become teal (8) with magenta holes (6).
    - For the right group, solid objects become green (3) and hollow objects become orange (7) with magenta holes (6).
    - After recoloring each object according to its group's palette, draw a one-cell 8-neighborhood red outline (2).

    Key insight: Local hole detection still matters, but the final colors cannot be chosen per object
    until the objects have been spatially clustered into the left and right groups.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    records = _component_records(g, bg)
    split = _group_split_index(records)
    groups = [(records[:split], LEFT_SCHEME), (records[split:], RIGHT_SCHEME)]

    for group_records, scheme in groups:
        for rec in group_records:
            mask = rec["mask"]
            holes = rec["holes"]
            out[mask] = int(scheme["hollow"] if holes else scheme["solid"])
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = int(scheme["hole"])

            outline = _outline_mask(rec["solid"])
            out[outline & (out == bg)] = 2

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (22, 28)
BG_COLOR = 4
FG_COLOR = 1

GROUP_COUNT = 2
COMPONENTS_PER_GROUP = 2
TOTAL_COMPONENTS = GROUP_COUNT * COMPONENTS_PER_GROUP

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 40

SMALL_SHAPE_PROB = 0.2
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

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300
GROUP_GAP_MIN = 7

GEN_OUTER_RETRIES = 400


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


def _fits_cluster(mask: np.ndarray) -> bool:
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
        return _make_small_shape()
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
    placed: list[tuple[int, int, np.ndarray, int]],
    group_id: int,
) -> bool:
    mh, mw = mask.shape
    if col < BORDER_MIN or col + mw > W - BORDER_MIN:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(BORDER_MIN, H - mh - BORDER_MIN)
        rows_a, cols_a = np.where(mask)
        rows_a, cols_a = rows_a + r, cols_a + col

        conflict = False
        for pr, pc, pm, _ in placed:
            row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
            col_gap = max(0, max(col - (pc + pm.shape[1] - 1), pc - (col + mw - 1)))
            if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                continue
            if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
                conflict = True
                break

        if not conflict:
            placed.append((r, col, mask, group_id))
            return True
    return False


def _intended_groups_match(grid: np.ndarray, placed: list[tuple[int, int, np.ndarray, int]]) -> bool:
    records = _component_records(grid, BG_COLOR)
    if len(records) != TOTAL_COMPONENTS:
        return False
    split = _group_split_index(records)
    if split != COMPONENTS_PER_GROUP:
        return False

    membership = []
    for rec in records:
        bbox = rec["bbox"]
        matched = None
        for r, c, mask, group_id in placed:
            mh, mw = mask.shape
            placed_bbox = (r, c, r + mh - 1, c + mw - 1)
            if placed_bbox == bbox:
                matched = group_id
                break
        membership.append(matched)

    return membership == [0, 0, 1, 1]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    group_components: list[list[np.ndarray]] = []
    for _ in range(GROUP_COUNT):
        hollow = None
        for _ in range(HOLLOW_GEN_RETRIES):
            hollow = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
            if hollow is not None and hollow.sum() >= HOLLOW_MIN_AREA:
                break
        if hollow is None or hollow.sum() < HOLLOW_MIN_AREA:
            return None

        solid = _make_solid_component()
        if solid is None:
            return None

        group_components.append([hollow, solid])

    left_max_w = max(mask.shape[1] for mask in group_components[0])
    right_max_w = max(mask.shape[1] for mask in group_components[1])
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
    placed: list[tuple[int, int, np.ndarray, int]] = []

    for group_id, (anchor, masks) in enumerate(zip([left_anchor, right_anchor], group_components)):
        for mask in sorted(masks, key=lambda arr: -arr.shape[0]):
            if not _place_component(H, W, mask, anchor, placed, group_id):
                return None

    for r, c, mask, _group_id in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    if not _intended_groups_match(grid, placed):
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
