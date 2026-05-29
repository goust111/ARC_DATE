# Combo variant of aa4ec2a5 (index 37)
# combo_spec: D1/1+D16/2
# Name: 条件轮廓+可变半径轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Base task: every blue 4-connected component gets a red Chebyshev-radius-1 outline; if
#      the component encloses a background hole, the body becomes teal and the hole magenta
#      before outlining.
#    - Combo task: first count the original blue area. Components with area <= 10 are skipped
#      entirely, so even hollow ones stay blue with yellow holes. Components with area 11-15
#      are processed with radius 2. Components with area >= 16 are processed with radius 3.
#    - Example: a 10-cell hollow ring is left unchanged; a 12-cell solid gets a radius-2 red
#      halo; a 20-cell hollow turns teal/magenta and gets a radius-3 halo. The difference from
#      the base task is therefore twofold: some objects are ignored, and the processed ones no
#      longer share one fixed outline width.
# b) Complexity calibration:
#    - The generator always includes three observable size regimes in one grid:
#      skipped (<= 10), narrow-halo (11-15), and wide-halo (>= 16).
#    - The area buckets are few and reused across samples, so the solver can infer a compact
#      rule instead of memorizing many unrelated cases.
# c) Implicit-assumption audit for main():
#    - The original loop assumed every component should be recolored and outlined. That is no
#      longer true, so the area filter must happen before hole recoloring.
#    - Radius must be computed from the original foreground area, not from the hole-filled mask,
#      otherwise hollow objects would encode a larger radius than equally sized solid objects.
#    - `out[outline & (out == bg)] = 2` is still required because wide halos must not overwrite
#      teal bodies, magenta holes, or neighboring objects.
# d) Placement legality:
#    - If two processed components have radii r1 and r2, their foreground cells need Chebyshev
#      distance at least r1 + r2 + 1 to keep the red halos disjoint.
#    - Skipped objects still keep a baseline separation of 3 cells for readability.
#    - Border clearance must be at least the active outline radius so the full halo stays visible.
# e) Visual observability:
#    - Every generated sample has at least one skipped object, one radius-2 object, and one
#      radius-3 object. A skipped hollow object is also forced in, making it obvious that
#      "contains a hole" alone does not trigger the transform.
# f) Visual cleanliness:
#    - The palette remains the same as the base task: blue input objects, red outlines, and
#      teal/magenta only for processed hollow objects. The generator limits object sizes and
#      spacing so wide halos do not merge into clutter.
# g) New assumptions audit:
#    - `main()` now assumes the intended training inputs contain objects on both sides of the
#      area threshold and at least two selected size buckets if the combo behavior is to be
#      learnable. `generate_input()` guarantees exactly that mix and rejects degenerate cases.


SKIP_MAX_AREA = 10
MEDIUM_MAX_AREA = 15

SKIPPED_RADIUS = 0
MEDIUM_RADIUS = 2
LARGE_RADIUS = 3


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _outline_radius(area: int) -> int:
    if area <= SKIP_MAX_AREA:
        return SKIPPED_RADIUS
    if area <= MEDIUM_MAX_AREA:
        return MEDIUM_RADIUS
    return LARGE_RADIUS


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
    if radius <= 0:
        return np.zeros_like(mask, dtype=bool)

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


def _base_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid, 1)
        out[outline & (out == bg)] = 2

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red outlines only around sufficiently large objects, with width determined by area.

    Input structure:
    - A uniform background contains several blue 4-connected foreground components.
    - Some components are hollow: inside their bounding boxes they enclose 4-connected
      background holes.
    - The components span multiple size bands, including some clearly below and above the
      outline threshold.

    Rule:
    - Count each component's original blue area.
    - Area <= 10: leave the component completely unchanged, even if it is hollow.
    - Area 11-15: apply the hollow recoloring rule if needed, then draw a red Chebyshev-radius-2
      outline on background cells only.
    - Area >= 16: apply the same hollow recoloring rule if needed, then draw a red
      Chebyshev-radius-3 outline on background cells only.

    Key insight: The solver must infer two linked conditions at once: only large-enough objects
    are processed at all, and among those selected objects the outline width still depends on
    the component's area.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        area = int(comp["area"])
        radius = _outline_radius(area)
        if radius == 0:
            continue

        mask = _coords_mask(g.shape, coords)
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

GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 27)
BG_COLOR = 4
FG_COLOR = 1

BASE_COMPONENT_GAP = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 700
GEN_OUTER_RETRIES = 400

SOLID_GEN_RETRIES = 120
SOLID_GROW_RETRIES = 30

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_HOLE_SEARCH_RETRIES = 250
HOLLOW_GEN_RETRIES = 160
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)

AREA_BUCKETS: dict[str, tuple[int, int]] = {
    "small": (6, 10),
    "medium": (11, 15),
    "large": (16, 26),
}

COMPONENT_TEMPLATES = (
    ("small_hollow", "medium_solid", "large_solid"),
    ("small_hollow", "medium_hollow", "large_solid"),
    ("small_hollow", "medium_solid", "large_hollow"),
    ("small_hollow", "medium_solid", "large_solid", "small_solid"),
    ("small_hollow", "medium_hollow", "large_solid", "small_solid"),
    ("small_hollow", "medium_solid", "large_hollow", "small_solid"),
)


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
            if remaining >= 2 and random.random() < 0.2:
                frontier2 = _frontier_candidates(mask)
                if frontier2:
                    best2 = max(score for score, _, _ in frontier2)
                    candidates2 = [(rr, cc) for score, rr, cc in frontier2 if score >= best2 - 1]
                    rr, cc = random.choice(candidates2)
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
        if target_area <= 5:
            return _make_exact_small_shape(target_area)
        mask = _grow_connected_shape(target_area)
        if mask is not None:
            return mask
    return None


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int, extension_prob: float) -> np.ndarray:
    if random.random() >= extension_prob:
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


def _make_hollow_component(
    target_hole_area: int,
    *,
    pad_range: tuple[int, int],
    max_notches: int,
    extension_prob: float,
) -> np.ndarray | None:
    dims = _rect_dims(target_hole_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*pad_range)
    pad_bot = random.randint(*pad_range)
    pad_left = random.randint(*pad_range)
    pad_right = random.randint(*pad_range)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, max_notches)
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        max_h = pad_top - 1 if corner[0] == "t" else pad_bot - 1
        max_w = pad_left - 1 if corner[1] == "l" else pad_right - 1
        if max_h <= 0 or max_w <= 0:
            continue
        nh = random.randint(1, max_h)
        nw = random.randint(1, max_w)
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, 0:nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    mask = _extend_mask(mask, base_h, base_w, extension_prob)
    return _carve_hole(mask, hh, hw)


def _make_hollow_component_in_area(
    area_lo: int,
    area_hi: int,
    *,
    hole_choices: list[int],
    pad_range: tuple[int, int],
    max_notches: int,
    extension_prob: float,
) -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        mask = _make_hollow_component(
            random.choice(hole_choices),
            pad_range=pad_range,
            max_notches=max_notches,
            extension_prob=extension_prob,
        )
        if mask is not None and area_lo <= int(mask.sum()) <= area_hi:
            return mask
    return None


def _build_component(kind: str) -> np.ndarray | None:
    if kind == "small_solid":
        return _make_solid_component(*AREA_BUCKETS["small"])
    if kind == "medium_solid":
        return _make_solid_component(*AREA_BUCKETS["medium"])
    if kind == "large_solid":
        return _make_solid_component(*AREA_BUCKETS["large"])
    if kind == "small_hollow":
        return _make_hollow_component_in_area(
            10,
            10,
            hole_choices=[2],
            pad_range=(1, 1),
            max_notches=0,
            extension_prob=0.0,
        )
    if kind == "medium_hollow":
        return _make_hollow_component_in_area(
            *AREA_BUCKETS["medium"],
            hole_choices=[2, 3, 4],
            pad_range=(1, 2),
            max_notches=1,
            extension_prob=0.2,
        )
    if kind == "large_hollow":
        return _make_hollow_component_in_area(
            *AREA_BUCKETS["large"],
            hole_choices=HOLE_VALID_AREAS,
            pad_range=(2, 3),
            max_notches=2,
            extension_prob=0.7,
        )
    raise ValueError(f"Unknown component kind: {kind}")


def _component_info(grid: np.ndarray) -> list[tuple[int, bool]]:
    info: list[tuple[int, bool]] = []
    for comp in find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        info.append((int(comp["area"]), bool(_interior_holes(mask, coords))))
    return info


def _gen_attempt() -> np.ndarray | None:
    kinds = list(random.choice(COMPONENT_TEMPLATES))
    random.shuffle(kinds)

    components: list[dict[str, object]] = []
    for kind in kinds:
        mask = _build_component(kind)
        if mask is None:
            return None
        area = int(mask.sum())
        radius = _outline_radius(area)
        components.append(
            {
                "kind": kind,
                "mask": mask,
                "area": area,
                "radius": radius,
            }
        )

    min_h = max(int(comp["mask"].shape[0]) + 2 * max(BORDER_MIN, int(comp["radius"])) for comp in components)
    min_w = max(int(comp["mask"].shape[1]) + 2 * max(BORDER_MIN, int(comp["radius"])) for comp in components)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    H = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    W = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])
    grid = np.full((H, W), BG_COLOR, dtype=int)

    placed: list[dict[str, object]] = []
    order = sorted(
        range(len(components)),
        key=lambda i: -(
            (components[i]["mask"].shape[0] + 2 * max(BORDER_MIN, int(components[i]["radius"])))
            * (components[i]["mask"].shape[1] + 2 * max(BORDER_MIN, int(components[i]["radius"])))
        ),
    )

    for idx in order:
        mask = components[idx]["mask"]
        radius = int(components[idx]["radius"])
        mh, mw = mask.shape
        clearance = max(BORDER_MIN, radius)
        row_min = clearance
        row_max = H - mh - clearance
        col_min = clearance
        col_max = W - mw - clearance
        if row_max < row_min or col_max < col_min:
            return None

        rows_local, cols_local = np.where(mask)
        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(row_min, row_max)
            c = random.randint(col_min, col_max)
            rows_abs = rows_local + r
            cols_abs = cols_local + c

            conflict = False
            for prev in placed:
                prev_rows = prev["rows_abs"]
                prev_cols = prev["cols_abs"]
                prev_r = int(prev["r"])
                prev_c = int(prev["c"])
                prev_mask = prev["mask"]
                prev_radius = int(prev["radius"])
                required_gap = max(BASE_COMPONENT_GAP, radius + prev_radius + 1)

                row_gap = max(0, max(r - (prev_r + prev_mask.shape[0] - 1), prev_r - (r + mh - 1)))
                col_gap = max(0, max(c - (prev_c + prev_mask.shape[1] - 1), prev_c - (c + mw - 1)))
                if max(row_gap, col_gap) >= required_gap:
                    continue
                if _chebyshev_conflict(rows_abs, cols_abs, prev_rows, prev_cols, required_gap):
                    conflict = True
                    break

            if not conflict:
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

    for item in placed:
        grid[item["rows_abs"], item["cols_abs"]] = FG_COLOR

    try:
        combo_out = main(grid)
        base_out = _base_main(grid)
    except Exception:
        return None

    info = _component_info(grid)
    areas = [area for area, _ in info]
    selected_radii = {_outline_radius(area) for area in areas if area > SKIP_MAX_AREA}

    if not any(area <= SKIP_MAX_AREA for area in areas):
        return None
    if not any(SKIP_MAX_AREA < area <= MEDIUM_MAX_AREA for area in areas):
        return None
    if not any(area > MEDIUM_MAX_AREA for area in areas):
        return None
    if selected_radii != {MEDIUM_RADIUS, LARGE_RADIUS}:
        return None
    if not any(area <= SKIP_MAX_AREA and has_hole for area, has_hole in info):
        return None
    if np.array_equal(combo_out, grid):
        return None
    if np.array_equal(combo_out, base_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
