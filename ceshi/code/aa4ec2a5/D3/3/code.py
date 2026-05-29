# Variant of aa4ec2a5
# Dimension: D3 — 源对象的连通性与形态 (src)
# Change: 蓝色4连通对象（可能有多个），部分对象内部封闭了背景色洞... → 嵌套对象（对象内部包含另一个颜色的对象而非洞）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the D3 variant
# ----------------------------------
# variant_spec:
#   input:
#     - Same canvas scale as the original task: yellow background with 3-4 separated objects.
#     - Every top-level object still has a blue outer shell.
#     - 1-2 of those blue shells are "nested": the shell encloses one interior pocket, but the
#       pocket is fully occupied by a small green inner component instead of yellow background.
#     - The green inner component fills the entire enclosed pocket, so the shell geometry is still
#       readable while the old "background hole" cue is removed.
#   output:
#     - Draw the same 1-cell red 8-neighborhood outline on background cells only.
#     - Solid blue objects stay blue.
#     - For nested objects, recolor only the outer blue shell to teal, keep the green inner object
#       unchanged, and outline the occupied union of shell + inner object.
#
# behavior_trace:
#   - Original local situation: blue shell surrounds a yellow hole -> output teal shell, magenta
#     hole fill, red outer outline.
#   - Variant local situation: the same shell surrounds a green inner object -> output teal shell,
#     green inner object preserved, red outer outline around the combined occupied region.
#   - Critical difference: the enclosed region is no longer an empty hole to be overwritten; it is
#     a real object whose identity survives the transform.
#
# complexity_calibration:
#   - Keep the original object count, shell irregularity, and placement spacing, so the puzzle still
#     looks like aa4ec2a5 rather than a different family.
#   - Restrict the new inner-object color to a single visible color (green=3). That keeps the rule
#     inferable from 2-4 examples while still making the D3 change obvious.
#   - Keep 1-2 nested examples per grid so the new branch is always represented, but not every object
#     uses it, preserving contrast against plain solid blue objects.
#
# implicit_assumption_audit:
#   - `_interior_regions` still works because the outer shell is detected from blue cells alone; the
#     enclosed pocket is defined by shell geometry, not by the pocket's fill color.
#   - `main()` only iterates blue components. This assumes every top-level object has a blue shell and
#     every non-blue component is nested inside one of those shells. `generate_input()` enforces that.
#   - Outline painting must still be restricted to background cells; otherwise preserved inner objects
#     would be overwritten by red when they touch the shell.
#   - The original code's "any enclosed non-shell region behaves like a hole" is no longer valid:
#     the enclosed region may already contain meaningful colored pixels, so `main()` must preserve them.
#
# placement_and_legality:
#   - Generation places whole occupied masks (outer shell plus any inner object), not just the blue
#     shell, so components cannot overlap or violate the intended spacing after nesting is added.
#   - Nested shells are produced from the original hollow-shell builder, then the enclosed pocket is
#     filled with green. This keeps the geometry legal for the solver while changing the semantics.
#
# observability_and_cleanliness:
#   - Humans can see the new rule because nested inputs contain a visible green core, and nested outputs
#     keep that core while recoloring only the shell. This is a direct visual contrast with solid objects.
#   - There is no palette explosion: blue/green in input, blue/teal/green/red in output.
#
# verification_metric:
#   - A generated sample is considered non-degenerate when it contains at least one blue shell with an
#     enclosed non-background pocket. Under the old solver the inner green object would be overwritten,
#     but under the variant solver it remains green, so the output must differ on those samples.


BG_COLOR = 4
FG_COLOR = 1
INNER_COLOR = 3
OUTLINE_COLOR = 2
OUTER_RECOLOR = 8


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _interior_regions(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    regions: list[np.ndarray] = []
    for region in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (region[:, 0] == 0)
            | (region[:, 1] == 0)
            | (region[:, 0] == sub.shape[0] - 1)
            | (region[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = region.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            regions.append(shifted)
    return regions


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


def _enclosed_occupied_regions(
    grid: np.ndarray, outer_mask: np.ndarray, coords: np.ndarray, bg: int
) -> list[np.ndarray]:
    occupied_regions: list[np.ndarray] = []
    for pocket in _interior_regions(outer_mask, coords):
        vals = grid[pocket[:, 0], pocket[:, 1]]
        occupied = pocket[vals != bg]
        if len(occupied) > 0:
            occupied_regions.append(occupied)
    return occupied_regions


def _count_nested_components(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    count = 0
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[FG_COLOR]):
        outer_mask = _coords_mask(g.shape, comp["coords"])
        if _enclosed_occupied_regions(g, outer_mask, comp["coords"], bg):
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around blue objects, preserving enclosed colored inner objects.

    Input structure:
    - A mostly uniform background contains one or more blue 4-connected outer components.
    - Some blue components enclose a fully internal pocket occupied by a smaller non-background object instead of background.

    Rule:
    - For every blue outer component, draw a one-cell 8-neighborhood outline in red (2).
    - If a blue component encloses another colored object, recolor only the blue shell to teal (8).
    - Keep the enclosed inner object in its original color, and compute the outline around the occupied union of shell and inner object.

    Key insight: The shell geometry still defines the enclosed pocket, but the pocket is no longer a hole to fill; it is an existing inner object that must be preserved while the shell is recolored.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[FG_COLOR]):
        coords = comp["coords"]
        outer_mask = _coords_mask(g.shape, coords)
        enclosed = _enclosed_occupied_regions(g, outer_mask, coords, bg)

        solid = outer_mask.copy()
        if enclosed:
            out[outer_mask] = OUTER_RECOLOR
            for region in enclosed:
                solid[region[:, 0], region[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

COMP_COUNT_RANGE = (3, 4)
NESTED_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

POCKET_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
POCKET_DIM_MAX = 4
NESTED_PAD_RANGE = (1, 3)
NESTED_MAX_NOTCHES = 2
NESTED_EXTENSION_PROB = 0.5
NESTED_EXT_H_RANGE = (1, 4)
NESTED_EXT_SIDE_W_RANGE = (1, 3)
NESTED_POCKET_SEARCH_RETRIES = 200
NESTED_GEN_RETRIES = 50
NESTED_MIN_OUTER_AREA = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

GEN_OUTER_RETRIES = 300


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _rect_dims(area: int) -> tuple[int, int] | None:
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= POCKET_DIM_MAX and area // i <= POCKET_DIM_MAX
    ]
    if not factors:
        return None
    h, w = random.choice(factors)
    if random.random() < 0.5:
        h, w = w, h
    return h, w


def _chebyshev_conflict(rows_a, cols_a, pr, pc, pm) -> bool:
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


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int) -> np.ndarray:
    if random.random() >= NESTED_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*NESTED_EXT_H_RANGE)

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
        ext_w_actual = random.randint(*NESTED_EXT_SIDE_W_RANGE)
        offset = random.randint(0, base_h - ext_h_actual)
        new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
        if side == "left":
            new_mask[:, ext_w_actual:] = mask
            new_mask[offset : offset + ext_h_actual, 0:ext_w_actual] = True
        else:
            new_mask[:, :base_w] = mask
            new_mask[offset : offset + ext_h_actual, base_w : base_w + ext_w_actual] = True

    return new_mask


def _carve_pocket(mask: np.ndarray, hh: int, hw: int) -> np.ndarray | None:
    mh, mw = mask.shape
    for _ in range(NESTED_POCKET_SEARCH_RETRIES):
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
        if cropped is None:
            continue
        pockets = _interior_regions(cropped, np.argwhere(cropped))
        if len(pockets) == 1:
            return cropped
    return None


def _make_nested_shell(target_pocket_area: int) -> np.ndarray | None:
    dims = _rect_dims(target_pocket_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*NESTED_PAD_RANGE)
    pad_bot = random.randint(*NESTED_PAD_RANGE)
    pad_left = random.randint(*NESTED_PAD_RANGE)
    pad_right = random.randint(*NESTED_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, NESTED_MAX_NOTCHES)
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
    return _carve_pocket(mask, hh, hw)


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


def _build_nested_component() -> dict[str, np.ndarray] | None:
    for _ in range(NESTED_GEN_RETRIES):
        outer = _make_nested_shell(random.choice(POCKET_VALID_AREAS))
        if outer is None or int(outer.sum()) < NESTED_MIN_OUTER_AREA:
            continue
        pockets = _interior_regions(outer, np.argwhere(outer))
        if len(pockets) != 1:
            continue
        inner = _coords_mask(outer.shape, pockets[0])
        occupied = outer | inner
        if not is_contiguous(occupied.astype(int), background=0, connectivity=4):
            continue
        return {"occupied": occupied, "outer": outer, "inner": inner}
    return None


def _build_solid_component() -> dict[str, np.ndarray] | None:
    if random.random() < SMALL_SHAPE_PROB:
        outer = _make_small_shape()
    else:
        outer = None
        for _ in range(POLY_GEN_RETRIES):
            outer = _make_polyomino()
            if outer is not None:
                break
        if outer is None:
            return None

    return {"occupied": outer.copy(), "outer": outer, "inner": np.zeros_like(outer, dtype=bool)}


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_nested = random.randint(
        NESTED_COUNT_RANGE[0], min(NESTED_COUNT_RANGE[1], num_components - 1)
    )

    indices = list(range(num_components))
    random.shuffle(indices)
    nested_indices = set(indices[:num_nested])

    components: list[dict[str, np.ndarray]] = []
    for i in range(num_components):
        comp = _build_nested_component() if i in nested_indices else _build_solid_component()
        if comp is None:
            return None
        components.append(comp)

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, dict[str, np.ndarray]]] = []

    order = sorted(range(num_components), key=lambda i: -int(components[i]["occupied"].sum()))

    for idx in order:
        comp = components[idx]
        occ = comp["occupied"]
        mh, mw = occ.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

            rows_a, cols_a = np.where(occ)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, prev in placed:
                prev_occ = prev["occupied"]
                row_gap = max(0, max(r - (pr + prev_occ.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + prev_occ.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, prev_occ):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, comp))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, comp in placed:
        outer_rows, outer_cols = np.where(comp["outer"])
        inner_rows, inner_cols = np.where(comp["inner"])
        grid[outer_rows + r, outer_cols + c] = FG_COLOR
        grid[inner_rows + r, inner_cols + c] = INNER_COLOR

    if _count_nested_components(grid) < 1:
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
