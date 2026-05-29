# Variant of aa4ec2a5
# Dimension: D12 — 颜色系统 (meta)
# Change: 使用4种颜色：背景(黑色)、对象(蓝色)、轮廓(红色)、有洞对象(青色)、洞填充(品红)，共5种颜色角色... → 使用图例定义颜色映射

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the D12 variant
# ----------------------------------
# variant_spec:
#   legend_geometry:
#     form: a 2x3 or 3x2 block placed flush against exactly one grid edge
#     orientation:
#       top/bottom: three vertical color-pairs
#       left/right: three horizontal color-pairs
#     semantic_slots:
#       slot_0: object_color -> outline_color
#       slot_1: object_color -> hollow_object_color
#       slot_2: hollow_object_color -> hole_fill_color
#     visual_pattern:
#       outer edge line repeats the object color twice, then shows the hollow color
#       inner line shows outline color, hollow color, hole-fill color
#     rationale:
#       the repeated/chained colors make the legend readable by position and by visible reuse,
#       without relying on hidden color ids or arbitrary ordering.
#   behavior_trace:
#     original:
#       objects are always one fixed color and the solver hard-codes output colors 2/8/6.
#     variant:
#       solver must first detect the edge legend, decode the four role colors, erase the
#       legend from the working mask, and only then run the same hole + outline geometry.
#     concrete_difference:
#       a hollow blue shape that used to become teal with magenta hole and red outline now
#       may become, for example, orange with grey hole and green outline if the legend says so.
#   complexity_calibration:
#     kept:
#       same object geometry, same 4-connected component detection, same hole definition,
#       same 8-neighborhood outline.
#     changed:
#       only the color system and the need to parse a legend, plus the placement buffer needed
#       to keep that legend visually separate.
#     not_over_simplified:
#       colors vary per sample and the legend side varies, so the new rule is visible.
#       every sample still contains multiple solid and hollow objects, giving repeated evidence.
#   implicit_assumption_audit:
#     - old `out[mask] = 8` assumed hollow recolor is globally fixed; now it must use the
#       legend's second target color.
#     - old `out[outline & (out == bg)] = 2` assumed the outline color is globally fixed; now
#       it must use the legend's first target color, and the legend area must already be
#       excluded so it is not outlined.
#     - old `find_components(g, bg=bg, ...)` was safe because every non-bg cell was a task
#       object; now the legend also uses non-bg cells, so the solver must replace the legend
#       area with bg before component finding.
#     - old fixed-color task tolerated objects near any edge; now the generator must reserve a
#       one-cell interior buffer next to the legend so outlines never collide with legend cells.
#   placement_legality:
#     generator still enforces component separation with Chebyshev distance >= 3 and border
#     clearance >= 1, while also reserving the legend block plus a one-cell moat.
#   visual_observability:
#     the legend is tiny, isolated, and sits on an edge. The repeated object color in slots 0/1
#     and repeated hollow color in slots 1/2 make the color transitions inferable from 2-4
#     examples without textual instructions.
#   cleanliness:
#     objects never touch the legend band, so legend, objects, holes, and outlines remain
#     visually distinct.


LEGEND_LEN = 3
LEGEND_THICKNESS = 2
LEGEND_SIDES = ("top", "bottom", "left", "right")


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


def _contiguous_run_start(vec: np.ndarray, bg: int, length: int) -> int | None:
    idx = np.flatnonzero(vec != bg)
    if len(idx) != length:
        return None
    start = int(idx[0])
    if not np.array_equal(idx, np.arange(start, start + length)):
        return None
    return start


def _legend_roles_from_lines(outer: np.ndarray, inner: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    outer_vals = [int(v) for v in outer.tolist()]
    inner_vals = [int(v) for v in inner.tolist()]
    if any(v == bg for v in outer_vals + inner_vals):
        return None
    if outer_vals[0] != outer_vals[1]:
        return None
    if outer_vals[2] != inner_vals[1]:
        return None
    object_color = outer_vals[0]
    outline_color = inner_vals[0]
    hollow_color = inner_vals[1]
    hole_color = inner_vals[2]
    if len({object_color, outline_color, hollow_color, hole_color}) != 4:
        return None
    return object_color, outline_color, hollow_color, hole_color


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[str, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[dict[str, int], np.ndarray]] = []

    if h >= 2 and w >= LEGEND_LEN:
        top0 = _contiguous_run_start(grid[0], bg, LEGEND_LEN)
        top1 = _contiguous_run_start(grid[1], bg, LEGEND_LEN)
        if top0 is not None and top0 == top1:
            roles = _legend_roles_from_lines(grid[0, top0 : top0 + LEGEND_LEN], grid[1, top0 : top0 + LEGEND_LEN], bg)
            if roles is not None:
                mask = np.zeros_like(grid, dtype=bool)
                mask[0:2, top0 : top0 + LEGEND_LEN] = True
                candidates.append((
                    {
                        "object": roles[0],
                        "outline": roles[1],
                        "hollow": roles[2],
                        "hole": roles[3],
                    },
                    mask,
                ))

        bot0 = _contiguous_run_start(grid[h - 1], bg, LEGEND_LEN)
        bot1 = _contiguous_run_start(grid[h - 2], bg, LEGEND_LEN)
        if bot0 is not None and bot0 == bot1:
            roles = _legend_roles_from_lines(grid[h - 1, bot0 : bot0 + LEGEND_LEN], grid[h - 2, bot0 : bot0 + LEGEND_LEN], bg)
            if roles is not None:
                mask = np.zeros_like(grid, dtype=bool)
                mask[h - 2 : h, bot0 : bot0 + LEGEND_LEN] = True
                candidates.append((
                    {
                        "object": roles[0],
                        "outline": roles[1],
                        "hollow": roles[2],
                        "hole": roles[3],
                    },
                    mask,
                ))

    if w >= 2 and h >= LEGEND_LEN:
        left0 = _contiguous_run_start(grid[:, 0], bg, LEGEND_LEN)
        left1 = _contiguous_run_start(grid[:, 1], bg, LEGEND_LEN)
        if left0 is not None and left0 == left1:
            roles = _legend_roles_from_lines(grid[left0 : left0 + LEGEND_LEN, 0], grid[left0 : left0 + LEGEND_LEN, 1], bg)
            if roles is not None:
                mask = np.zeros_like(grid, dtype=bool)
                mask[left0 : left0 + LEGEND_LEN, 0:2] = True
                candidates.append((
                    {
                        "object": roles[0],
                        "outline": roles[1],
                        "hollow": roles[2],
                        "hole": roles[3],
                    },
                    mask,
                ))

        right0 = _contiguous_run_start(grid[:, w - 1], bg, LEGEND_LEN)
        right1 = _contiguous_run_start(grid[:, w - 2], bg, LEGEND_LEN)
        if right0 is not None and right0 == right1:
            roles = _legend_roles_from_lines(grid[right0 : right0 + LEGEND_LEN, w - 1], grid[right0 : right0 + LEGEND_LEN, w - 2], bg)
            if roles is not None:
                mask = np.zeros_like(grid, dtype=bool)
                mask[right0 : right0 + LEGEND_LEN, w - 2 : w] = True
                candidates.append((
                    {
                        "object": roles[0],
                        "outline": roles[1],
                        "hollow": roles[2],
                        "hole": roles[3],
                    },
                    mask,
                ))

    assert len(candidates) == 1, "legend must appear exactly once on a grid edge"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use an edge legend to decide how objects, holes, and outlines should be recolored.

    Input structure:
    - A mostly uniform background contains several 4-connected objects, all drawn in one shared source color.
    - A small legend block sits on one grid edge and shows three ordered color pairs that encode the task palette.

    Rule:
    - Read the legend as: object->outline, object->hollow-object recolor, hollow-object recolor->hole-fill.
    - Ignore the legend while finding foreground components; for every object, draw a one-cell 8-neighborhood outline using the decoded outline color.
    - If an object encloses background holes, recolor the object body with the decoded hollow color, fill its holes with the decoded hole color, then compute the outline around that solidified mask.

    Key insight: The geometry is unchanged from the base task; the only new challenge is that the four functional colors are no longer fixed constants and must be inferred from the legend's chained color pairs.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    roles, legend_mask = _extract_legend(g, bg)

    work = g.copy()
    work[legend_mask] = bg
    out = g.copy()

    for comp in find_components(work, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = roles["hollow"]
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = roles["hole"]
                solid[hole[:, 0], hole[:, 1]] = True
        else:
            out[mask] = roles["object"]

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = roles["outline"]

    return out


# --------------- generator (c) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

# Component counts
COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

# Polyomino construction (solid shapes)
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

# Small solid shapes
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Hollow component construction
HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

# Placement
CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

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


def _chebyshev_conflict(rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a, strict=False):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _make_polyomino() -> np.ndarray | None:
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [(random.randint(*POLY_RECT_DIM_RANGE), random.randint(*POLY_RECT_DIM_RANGE)) for _ in range(num_rects)]

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


def _choose_colors() -> tuple[int, int, int, int, int]:
    colors = list(range(10))
    random.shuffle(colors)
    bg, object_color, outline_color, hollow_color, hole_color = colors[:5]
    return bg, object_color, outline_color, hollow_color, hole_color


def _build_legend(side: str, object_color: int, outline_color: int, hollow_color: int, hole_color: int) -> np.ndarray:
    if side == "top":
        return np.array(
            [
                [object_color, object_color, hollow_color],
                [outline_color, hollow_color, hole_color],
            ],
            dtype=int,
        )
    if side == "bottom":
        return np.array(
            [
                [outline_color, hollow_color, hole_color],
                [object_color, object_color, hollow_color],
            ],
            dtype=int,
        )
    if side == "left":
        return np.array(
            [
                [object_color, outline_color],
                [object_color, hollow_color],
                [hollow_color, hole_color],
            ],
            dtype=int,
        )
    return np.array(
        [
            [outline_color, object_color],
            [hollow_color, object_color],
            [hole_color, hollow_color],
        ],
        dtype=int,
    )


def _place_legend(grid: np.ndarray, reserved: np.ndarray, legend: np.ndarray, side: str) -> np.ndarray:
    h, w = grid.shape
    lh, lw = legend.shape
    legend_mask = np.zeros_like(grid, dtype=bool)

    if side == "top":
        r0, c0 = 0, random.randint(0, w - lw)
    elif side == "bottom":
        r0, c0 = h - lh, random.randint(0, w - lw)
    elif side == "left":
        r0, c0 = random.randint(0, h - lh), 0
    else:
        r0, c0 = random.randint(0, h - lh), w - lw

    grid[r0 : r0 + lh, c0 : c0 + lw] = legend
    legend_mask[r0 : r0 + lh, c0 : c0 + lw] = True
    reserved[max(0, r0 - 1) : min(h, r0 + lh + 1), max(0, c0 - 1) : min(w, c0 + lw + 1)] = True
    return legend_mask


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg, object_color, outline_color, hollow_color, hole_color = _choose_colors()
    side = random.choice(LEGEND_SIDES)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in hollow_indices:
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

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    legend = _build_legend(side, object_color, outline_color, hollow_color, hole_color)
    _place_legend(grid, reserved, legend, side)

    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        rows_local, cols_local = np.where(mask)

        placed_ok = False
        r_hi = h - mh - BORDER_MIN
        c_hi = w - mw - BORDER_MIN
        if r_hi < BORDER_MIN or c_hi < BORDER_MIN:
            return None

        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, r_hi)
            c = random.randint(BORDER_MIN, c_hi)
            if r + mh > h - BORDER_MIN or c + mw > w - BORDER_MIN:
                continue
            if reserved[r : r + mh, c : c + mw][mask].any():
                continue

            rows_a, cols_a = rows_local + r, cols_local + c

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
        grid[rows + r, cols + c] = object_color

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
