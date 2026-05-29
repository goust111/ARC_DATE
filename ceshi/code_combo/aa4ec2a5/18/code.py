# Combo variant of aa4ec2a5 (index 18)
# combo_spec: D1/1+D6/0
# Name: 输出扩展+可变半径轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# ---------------
# combo_analysis:
#   fused_rule:
#     - Keep the original hollow-object recoloring rule.
#     - Replace the fixed 1-pixel Chebyshev outline with an area-coded radius:
#       area 1-5 -> radius 1, area 6-15 -> radius 2, area >= 16 -> radius 3.
#     - Because the widest outline may now be 2 or 3 pixels, the output canvas must be
#       expanded by `max_radius` on all four sides before any recoloring or outlining.
#   a_specific_behavior_and_trace:
#     - Original task: a blue object near the border is outlined with radius 1 directly on the
#       input canvas, so any outward pixels beyond the border are clipped.
#     - Fused task: first count each component's blue-cell area, map it to radius 1/2/3, then
#       find the largest radius among all components and add that much yellow padding everywhere.
#       After the shift onto the larger canvas, hollow components still become teal with magenta
#       holes, and each component gets its own red Chebyshev halo using its own radius.
#     - Concrete trace: suppose the input has a 4-cell solid touching the left edge, a 12-cell
#       hollow object, and a 20-cell solid touching the top edge. Their radii are 1, 2, and 3.
#       The output therefore expands by 3 cells on every side. The 4-cell object keeps blue and
#       gets a thin halo, the 12-cell hollow becomes teal/magenta with a medium halo, and the
#       20-cell solid gets a thick halo that now reaches the new outer frame instead of clipping.
#   b_complexity_calibration:
#     - The radius mapping uses only three buckets, which is enough to create a visible multi-way
#       size rule without becoming hard to infer from a few examples.
#     - The generator alternates between samples whose maximum radius is 2 and 3, so output-size
#       growth is not accidentally constant across the dataset.
#     - Each sample also contains at least one small object, ensuring the radius-1 case remains
#       visible even when the global padding is driven by a larger component.
#   c_hidden_assumption_audit:
#     - Original `out = g.copy()` assumed output shape equals input shape; the combo breaks that,
#       so the solver must allocate a larger canvas first.
#     - Original outlining assumed clipping at the old border was acceptable; after fusion it is
#       not, because the required pad depends on the largest inferred radius.
#     - Radius must be computed from the original blue foreground area, not from a hole-filled mask,
#       otherwise hollow objects would encode the wrong width.
#     - `out[outline & (out == bg)] = 2` remains valid only if the generator keeps enough pairwise
#       spacing that different halos never need to overwrite one another.
#   d_placement_legality:
#     - If two components have radii r1 and r2, their foreground cells must be at Chebyshev
#       distance at least `r1 + r2 + 1`; otherwise the red halos would collide.
#     - Input border clearance can now be 0 because the solver adds a margin of `max_radius`.
#     - The generator forces at least one component with radius `max_radius` to touch an input
#       edge, so the new outer padding is behaviorally necessary and not just empty space.
#   e_visual_observability:
#     - A solver can see two linked regularities: bigger objects have thicker red halos, and the
#       total output growth equals twice the thickest halo present in that input.
#     - Samples with `max_radius = 2` and samples with `max_radius = 3` are both produced, so the
#       output-size rule is supported by repeated but non-constant evidence.
#   f_visual_cleanliness:
#     - Colors stay close to the base task: blue for ordinary objects, teal/magenta for hollow
#       objects in the output, red for outlines, yellow for background.
#     - Large spacing is preserved so wide halos remain disjoint and readable.
#   g_color_and_rule_inference:
#     - No new color branch is introduced; the only new inference burden is the chained size rule
#       `area -> radius -> global padding`.
#     - Hollow recoloring still has a strong structural cue because magenta appears only in fully
#       enclosed holes inside teal bodies.
#   new_main_assumptions_to_check:
#     - The input must still use one dominant background color and blue foreground components.
#     - Components must remain 4-connected so area and hole detection are stable.
#     - The generator must keep at least one radius-`max_radius` object on the input border so the
#       expanded frame actually receives red pixels and exposes the D6 behavior.


SMALL_MAX_AREA = 5
MEDIUM_MAX_AREA = 15


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _outline_radius(area: int) -> int:
    if area <= SMALL_MAX_AREA:
        return 1
    if area <= MEDIUM_MAX_AREA:
        return 2
    return 3


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


def _outline_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _expand_canvas(grid: np.ndarray, bg: int, pad: int) -> np.ndarray:
    if pad <= 0:
        return np.array(grid, copy=True)
    h, w = grid.shape
    out = np.full((h + 2 * pad, w + 2 * pad), int(bg), dtype=int)
    out[pad : pad + h, pad : pad + w] = grid
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the output canvas by the widest required halo, then draw size-dependent outlines.

    Input structure:
    - A uniform background contains several blue 4-connected foreground components.
    - Some components enclose 4-connected background holes inside their bounding boxes, and
      components may touch the input border because the final halo may need space outside it.

    Rule:
    - Measure each component's original blue area and map it to a halo radius:
      area <= 5 -> 1, area 6-15 -> 2, area > 15 -> 3.
    - Let `pad` be the largest such radius in the grid; create an output canvas expanded by
      `pad` cells on every side and place the whole input in its center.
    - On that expanded canvas, recolor hollow components to teal (8) and their enclosed holes
      to magenta (6), then draw each component's red (2) Chebyshev-neighborhood outline using
      its own radius, writing only onto background cells.

    Key insight: The transformation now has a two-step size dependency: each object's area picks
    its local outline width, and the largest local width determines the global amount of padding.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    if not comps:
        return g.copy()

    pad = max(_outline_radius(int(comp["area"])) for comp in comps)
    expanded = _expand_canvas(g, bg, pad)
    out = expanded.copy()

    for comp in find_components(expanded, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        area = int(comp["area"])
        radius = _outline_radius(area)
        mask = _coords_mask(expanded.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid, radius)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator ---------------

# Canvas / grid bounds
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 27)
BG_COLOR = 4
FG_COLOR = 1

# Area buckets used by the fused transform
AREA_BUCKETS: dict[str, tuple[int, int]] = {
    "small": (1, 5),
    "medium": (6, 15),
    "large": (16, 28),
}
HOLLOW_AREA_BUCKETS: dict[str, tuple[int, int]] = {
    "medium": (10, 15),
    "large": (16, 28),
}

# Component counts
COMP_COUNT_RANGE = (3, 4)

# Solid component generation
SOLID_GEN_RETRIES = 80
SOLID_GROW_RETRIES = 20

# Hollow component construction
HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 120

# Placement
PLACEMENT_RETRIES = 700

# Top-level generation
GEN_OUTER_RETRIES = 300


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
        if not touches_edge:
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


def _chebyshev_conflict(
    rows_a: np.ndarray,
    cols_a: np.ndarray,
    rows_b: np.ndarray,
    cols_b: np.ndarray,
    min_dist: int,
) -> bool:
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < min_dist:
            return True
    return False


def _make_exact_small_shape(area: int) -> np.ndarray:
    options: dict[int, list[np.ndarray]] = {
        1: [np.array([[1]], dtype=bool)],
        2: [np.array([[1, 1]], dtype=bool), np.array([[1], [1]], dtype=bool)],
        3: [
            np.array([[1, 1, 1]], dtype=bool),
            np.array([[1], [1], [1]], dtype=bool),
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.array([[1, 0], [1, 1]], dtype=bool),
        ],
        4: [
            np.array([[1, 1], [1, 1]], dtype=bool),
            np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
            np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
            np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
            np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
        ],
        5: [
            np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
            np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
            np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
            np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
            np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
        ],
    }
    return random.choice(options[area]).copy()


def _frontier_candidates(mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = mask.shape
    frontier: dict[tuple[int, int], int] = {}
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                score = 0
                for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                    if 0 <= ar < h and 0 <= ac < w and mask[ar, ac]:
                        score += 1
                frontier[(nr, nc)] = max(frontier.get((nr, nc), 0), score)
    return [(score, r, c) for (r, c), score in frontier.items()]


def _grow_connected_shape(target_area: int) -> np.ndarray | None:
    canvas = max(9, int(np.ceil(np.sqrt(target_area))) + 7)
    for _ in range(SOLID_GROW_RETRIES):
        mask = np.zeros((canvas, canvas), dtype=bool)
        center = canvas // 2
        mask[center, center] = True

        while int(mask.sum()) < target_area:
            frontier = _frontier_candidates(mask)
            if not frontier:
                break
            if random.random() < 0.75:
                best = max(score for score, _, _ in frontier)
                candidates = [(r, c) for score, r, c in frontier if score == best]
            else:
                candidates = [(r, c) for _, r, c in frontier]
            r, c = random.choice(candidates)
            mask[r, c] = True

            remaining = target_area - int(mask.sum())
            if remaining >= 2 and random.random() < 0.20:
                second_frontier = _frontier_candidates(mask)
                if second_frontier:
                    second_best = max(score for score, _, _ in second_frontier)
                    second_candidates = [
                        (rr, cc) for score, rr, cc in second_frontier if score >= second_best - 1
                    ]
                    rr, cc = random.choice(second_candidates)
                    mask[rr, cc] = True

        cropped = _crop_mask(mask)
        if cropped is None or int(cropped.sum()) != target_area:
            continue
        if target_area >= 6 and min(cropped.shape) == 1:
            continue
        if target_area >= 10 and cropped.shape[0] * cropped.shape[1] > target_area * 4:
            continue
        if is_contiguous(cropped.astype(int), background=0, connectivity=4):
            return cropped
    return None


def _make_solid_component(area_lo: int, area_hi: int) -> np.ndarray | None:
    for _ in range(SOLID_GEN_RETRIES):
        target_area = random.randint(area_lo, area_hi)
        if target_area <= SMALL_MAX_AREA:
            return _make_exact_small_shape(target_area)
        mask = _grow_connected_shape(target_area)
        if mask is not None:
            return mask
    return None


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
        if cropped is not None and _has_interior_hole(cropped):
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
        nh = random.randint(
            1, max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1)
        )
        nw = random.randint(
            1, max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1)
        )
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


def _make_hollow_component_in_area(area_lo: int, area_hi: int) -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
        if mask is not None and area_lo <= int(mask.sum()) <= area_hi:
            return mask
    return None


def _component_plan(max_radius: int, num_components: int) -> list[dict[str, object]]:
    if max_radius == 2:
        plan = [
            {"bucket": "small", "hollow": False},
            {"bucket": "medium", "hollow": True},
            {"bucket": "medium", "hollow": False},
        ]
        if num_components == 4:
            extra_bucket = random.choice(["small", "medium"])
            extra_hollow = extra_bucket == "medium" and random.random() < 0.55
            plan.append({"bucket": extra_bucket, "hollow": extra_hollow})
    else:
        plan = [
            {"bucket": "small", "hollow": False},
            {"bucket": "medium", "hollow": True},
            {"bucket": "large", "hollow": False},
        ]
        if num_components == 4:
            extra_bucket = random.choices(["small", "medium", "large"], weights=[2, 2, 1])[0]
            extra_hollow = extra_bucket != "small" and random.random() < 0.4
            plan.append({"bucket": extra_bucket, "hollow": extra_hollow})

    if sum(1 for item in plan if bool(item["hollow"])) > 2:
        for item in reversed(plan):
            if bool(item["hollow"]):
                item["hollow"] = False
                break
    random.shuffle(plan)
    return plan


def _sample_edge_position(H: int, W: int, mh: int, mw: int) -> tuple[int, int]:
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return 0, random.randint(0, W - mw)
    if side == "bottom":
        return H - mh, random.randint(0, W - mw)
    if side == "left":
        return random.randint(0, H - mh), 0
    return random.randint(0, H - mh), W - mw


def _touches_grid_edge(rows_abs: np.ndarray, cols_abs: np.ndarray, H: int, W: int) -> bool:
    return bool(
        np.any(rows_abs == 0)
        or np.any(rows_abs == H - 1)
        or np.any(cols_abs == 0)
        or np.any(cols_abs == W - 1)
    )


def _placement_conflict(
    r: int,
    c: int,
    mask: np.ndarray,
    radius: int,
    placed: list[dict[str, object]],
) -> tuple[bool, np.ndarray, np.ndarray]:
    rows_local, cols_local = np.where(mask)
    rows_abs = rows_local + r
    cols_abs = cols_local + c

    mh, mw = mask.shape
    for prev in placed:
        prev_rows = prev["rows_abs"]
        prev_cols = prev["cols_abs"]
        prev_r = int(prev["r"])
        prev_c = int(prev["c"])
        prev_mh, prev_mw = prev["mask"].shape
        required_gap = radius + int(prev["radius"]) + 1

        row_gap = max(0, max(r - (prev_r + prev_mh - 1), prev_r - (r + mh - 1)))
        col_gap = max(0, max(c - (prev_c + prev_mw - 1), prev_c - (c + mw - 1)))
        if max(row_gap, col_gap) >= required_gap:
            continue
        if _chebyshev_conflict(rows_abs, cols_abs, prev_rows, prev_cols, required_gap):
            return True, rows_abs, cols_abs

    return False, rows_abs, cols_abs


def _gen_attempt() -> np.ndarray | None:
    max_radius = random.choice([2, 3])
    num_components = random.randint(*COMP_COUNT_RANGE)
    plan = _component_plan(max_radius, num_components)

    components: list[dict[str, object]] = []
    for item in plan:
        bucket = str(item["bucket"])
        hollow = bool(item["hollow"])
        if hollow:
            area_lo, area_hi = HOLLOW_AREA_BUCKETS[bucket]
            mask = _make_hollow_component_in_area(area_lo, area_hi)
        else:
            area_lo, area_hi = AREA_BUCKETS[bucket]
            mask = _make_solid_component(area_lo, area_hi)
        if mask is None:
            return None

        area = int(mask.sum())
        radius = _outline_radius(area)
        if radius > max_radius:
            return None
        components.append(
            {
                "mask": mask,
                "area": area,
                "radius": radius,
                "bucket": bucket,
                "hollow": hollow,
            }
        )

    actual_radii = [int(comp["radius"]) for comp in components]
    if max(actual_radii) != max_radius or len(set(actual_radii)) < 2:
        return None

    min_h = max(int(comp["mask"].shape[0]) for comp in components)
    min_w = max(int(comp["mask"].shape[1]) for comp in components)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    H = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    W = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[dict[str, object]] = []

    max_radius_indices = [i for i, comp in enumerate(components) if int(comp["radius"]) == max_radius]
    edge_idx = max(max_radius_indices, key=lambda i: int(components[i]["area"]))

    order = [edge_idx] + sorted(
        [i for i in range(num_components) if i != edge_idx],
        key=lambda i: -(
            (components[i]["mask"].shape[0] + 2 * int(components[i]["radius"]))
            * (components[i]["mask"].shape[1] + 2 * int(components[i]["radius"]))
        ),
    )

    for idx in order:
        mask = components[idx]["mask"]
        radius = int(components[idx]["radius"])
        mh, mw = mask.shape
        if mh > H or mw > W:
            return None

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            if idx == edge_idx:
                r, c = _sample_edge_position(H, W, mh, mw)
            else:
                r = random.randint(0, H - mh)
                c = random.randint(0, W - mw)

            conflict, rows_abs, cols_abs = _placement_conflict(r, c, mask, radius, placed)
            if conflict:
                continue

            placed.append(
                {
                    "r": r,
                    "c": c,
                    "mask": mask,
                    "rows_abs": rows_abs,
                    "cols_abs": cols_abs,
                    "radius": radius,
                }
            )
            placed_ok = True
            break

        if not placed_ok:
            return None

    touches_max_edge = False
    for item in placed:
        grid[item["rows_abs"], item["cols_abs"]] = FG_COLOR
        if int(item["radius"]) == max_radius and _touches_grid_edge(
            item["rows_abs"], item["cols_abs"], H, W
        ):
            touches_max_edge = True

    if not touches_max_edge:
        return None

    out = main(grid)
    if out.shape != (H + 2 * max_radius, W + 2 * max_radius):
        return None

    outer_frame = np.concatenate([out[0, :], out[-1, :], out[1:-1, 0], out[1:-1, -1]])
    if not np.any(outer_frame == 2):
        return None

    comps = find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True)
    observed_radii = sorted(_outline_radius(int(comp["area"])) for comp in comps)
    if max(observed_radii) != max_radius or len(set(observed_radii)) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
