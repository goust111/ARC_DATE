# Combo variant of aa4ec2a5 (index 28)
# combo_spec: D9/1+D16/0
# Name: 多对象合并统一轮廓+相邻对象共享轮廓

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for D9/1 + D16/0
# a) Concrete trace:
#    Original task treats each blue 4-connected object independently:
#    recolor hollow objects, then draw one red 8-neighborhood outline per object.
#    Example: two solid objects with a one-cell gap each get two small outlines, and the
#    gap cell becomes red. In this combo, objects whose minimum Chebyshev distance is <= 2
#    are first grouped. Their filled solids are unioned, a radius-1 8-neighborhood closing
#    bridges the short gap, and only one outer red contour is drawn for the group. The gap
#    between grouped objects stops being outlined, so the output visibly differs.
# b) Complexity calibration:
#    The grouping rule uses one discrete threshold only (distance <= 2), which is easy to
#    observe across 2-4 ARC examples. Samples will always contain at least one merged group
#    plus at least one isolated object, so solvers can contrast "shared outer contour" with
#    "ordinary single-object contour" inside the same puzzle family.
# c) Hidden assumptions broken by the combo:
#    - Original main() assumes 1 component -> 1 outline. This is replaced by component-group
#      processing.
#    - Original outline suppression used only the component's own mask. Now short gaps between
#      grouped objects must also suppress red, so the suppressing mask is the closed group mask.
#    - Original generator forced Chebyshev >= 3, which made outline overlap impossible and
#      would completely hide this combo. The new generator deliberately creates <= 2 spacing
#      inside one cluster and >= 4 spacing elsewhere.
# d) Placement legality:
#    Components are still non-overlapping, stay inside the canvas, and only the intended
#    cluster members are allowed within Chebyshev distance 1-2. All other pairs remain >= 4.
# e/f/g) Visual observability and cleanliness:
#    The only new visual effect is red contour fusion. Object colors remain the original
#    blue / teal / magenta scheme, so the extra reasoning burden is localized to spatial
#    grouping rather than a new color code. Because the generator avoids dense all-object
#    collisions, the fused contour stays legible instead of becoming cluttered.


GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 27)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 60
POLY_GEN_RETRIES = 40

SMALL_SHAPE_PROB = 0.25
SMALL_SHAPE_SIZE_RANGE = (1, 4)

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 250
HOLLOW_GEN_RETRIES = 60
HOLLOW_MIN_AREA = 10

BORDER_MIN = 1
MERGE_DISTANCE_CHOICES = (1, 2)
ISOLATED_MIN_CHEBYSHEV = 4
PLACEMENT_RETRIES = 500
GEN_OUTER_RETRIES = 400

STRUCT8 = np.ones((3, 3), dtype=bool)


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


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
    dilated = ndimage.binary_dilation(mask, structure=STRUCT8)
    return dilated & ~mask


def _component_records(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    records: list[dict] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        records.append(
            {
                "coords": coords,
                "mask": mask,
                "solid": solid,
                "holes": holes,
            }
        )
    return bg, records


def _group_components(records: list[dict]) -> list[list[int]]:
    n = len(records)
    if n == 0:
        return []

    dilated = [ndimage.binary_dilation(rec["mask"], structure=STRUCT8) for rec in records]
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if bool(np.any(dilated[i] & dilated[j])):
                graph[i].append(j)
                graph[j].append(i)

    groups: list[list[int]] = []
    seen = [False] * n
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        group: list[int] = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in graph[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        groups.append(sorted(group))
    return groups


def _merged_group_mask(records: list[dict], group: list[int], shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=bool)
    for idx in group:
        merged |= records[idx]["solid"]
    if len(group) > 1:
        merged = ndimage.binary_closing(merged, structure=STRUCT8)
    return merged


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, records = _component_records(g)
    out = g.copy()

    for rec in records:
        if rec["holes"]:
            out[rec["mask"]] = 8
            for hole in rec["holes"]:
                out[hole[:, 0], hole[:, 1]] = 6
        outline = _outline_mask(rec["solid"])
        out[outline & (out == bg)] = 2
    return out


def _fusion_metrics(grid: np.ndarray) -> dict:
    bg, records = _component_records(grid)
    fused = main(grid)
    baseline = _baseline_main(grid)
    group_sizes = sorted(len(group) for group in _group_components(records))
    return {
        "bg": bg,
        "component_count": len(records),
        "group_sizes": group_sizes,
        "largest_group": max(group_sizes, default=0),
        "red_delta": int(np.sum(baseline == 2) - np.sum(fused == 2)),
        "diff_cells": int(np.sum(fused != baseline)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects as before, but merge nearby objects into shared outline groups.

    Input structure:
    - A yellow background contains several blue 4-connected components.
    - Some components enclose background holes inside their own bounding boxes.
    - Distinct components may now sit within Chebyshev distance 1-2 of each other.

    Rule:
    - Any component with enclosed holes is recolored to teal (8), and its hole cells become
      magenta (6) before any outline is drawn.
    - Components whose minimum Chebyshev distance is at most 2 are treated as one outline
      group rather than separate outline targets.
    - For each group, form the union of all member solids (with holes filled), bridge the
      short inter-object gaps via a one-step 8-neighborhood closing, and draw one red (2)
      outer contour only on background cells.

    Key insight:
    - The recoloring rule is still object-local, but the red border rule is group-local:
      nearby objects suppress the red gap between them and share one larger enclosing outline.
    """
    g = np.asarray(grid, dtype=int)
    bg, records = _component_records(g)
    out = g.copy()

    for rec in records:
        if rec["holes"]:
            out[rec["mask"]] = 8
            for hole in rec["holes"]:
                out[hole[:, 0], hole[:, 1]] = 6

    for group in _group_components(records):
        merged = _merged_group_mask(records, group, g.shape)
        outline = _outline_mask(merged)
        out[outline & (out == bg)] = 2

    return out


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

    for rh, rw in rects[1:]:
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


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        return (
            np.ones((1, 2), dtype=bool)
            if random.random() < 0.5
            else np.ones((2, 1), dtype=bool)
        )
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


def _shape_coords(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    return rows.astype(int), cols.astype(int)


def _min_chebyshev_distance(
    rows_a: np.ndarray,
    cols_a: np.ndarray,
    rows_b: np.ndarray,
    cols_b: np.ndarray,
) -> int:
    dist = np.maximum(
        np.abs(rows_a[:, None] - rows_b[None, :]),
        np.abs(cols_a[:, None] - cols_b[None, :]),
    )
    return int(dist.min())


def _placed_distances(
    rows: np.ndarray,
    cols: np.ndarray,
    placed: list[dict],
) -> list[int]:
    out: list[int] = []
    for item in placed:
        out.append(_min_chebyshev_distance(rows, cols, item["rows"], item["cols"]))
    return out


def _place_component(
    H: int,
    W: int,
    mask: np.ndarray,
    placed: list[dict],
    *,
    need_neighbor_merge: bool,
    need_far_from_all: bool,
) -> dict | None:
    mh, mw = mask.shape
    rel_rows, rel_cols = _shape_coords(mask)

    r_max = H - mh - BORDER_MIN
    c_max = W - mw - BORDER_MIN
    if r_max < BORDER_MIN or c_max < BORDER_MIN:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(BORDER_MIN, r_max)
        c0 = random.randint(BORDER_MIN, c_max)
        rows = rel_rows + r0
        cols = rel_cols + c0

        if not placed:
            if need_neighbor_merge or need_far_from_all:
                continue
            return {"r": r0, "c": c0, "mask": mask, "rows": rows, "cols": cols}

        dists = _placed_distances(rows, cols, placed)
        if min(dists) <= 0:
            continue

        if need_neighbor_merge:
            if not any(dist in MERGE_DISTANCE_CHOICES for dist in dists):
                continue
            if any(dist < 1 for dist in dists):
                continue
        elif need_far_from_all:
            if min(dists) < ISOLATED_MIN_CHEBYSHEV:
                continue

        return {"r": r0, "c": c0, "mask": mask, "rows": rows, "cols": cols}
    return None


def _render_grid(H: int, W: int, placed: list[dict]) -> np.ndarray:
    grid = np.full((H, W), BG_COLOR, dtype=int)
    for item in placed:
        grid[item["rows"], item["cols"]] = FG_COLOR
    return grid


def _gen_components(num_components: int, hollow_indices: set[int]) -> list[np.ndarray] | None:
    components: list[np.ndarray] = []
    for idx in range(num_components):
        if idx in hollow_indices:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
                    components.append(mask)
                    found = True
                    break
            if not found:
                return None
        elif random.random() < SMALL_SHAPE_PROB:
            components.append(_make_small_shape())
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append(mask)
                    break
            else:
                return None
    return components


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])

    components = _gen_components(num_components, hollow_indices)
    if components is None:
        return None

    merge_group_size = 2 if num_components == 3 else random.choice([2, 3])
    merge_group = set(random.sample(range(num_components), merge_group_size))
    grouped_order = sorted(merge_group, key=lambda i: -int(components[i].sum()))
    isolated_order = sorted(
        [i for i in range(num_components) if i not in merge_group],
        key=lambda i: -int(components[i].sum()),
    )

    placed: list[dict] = []
    for k, idx in enumerate(grouped_order):
        placement = _place_component(
            H,
            W,
            components[idx],
            placed,
            need_neighbor_merge=(k > 0),
            need_far_from_all=False,
        )
        if placement is None:
            return None
        placed.append(placement)

    for idx in isolated_order:
        placement = _place_component(
            H,
            W,
            components[idx],
            placed,
            need_neighbor_merge=False,
            need_far_from_all=True,
        )
        if placement is None:
            return None
        placed.append(placement)

    grid = _render_grid(H, W, placed)

    metrics = _fusion_metrics(grid)
    if metrics["largest_group"] < 2:
        return None
    if metrics["red_delta"] <= 0 or metrics["diff_cells"] <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
