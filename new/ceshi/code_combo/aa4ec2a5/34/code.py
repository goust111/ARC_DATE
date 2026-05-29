# Combo variant of aa4ec2a5 (index 34)
# combo_spec: D3/3+D5/3+D10/2
# Name: 嵌套对象+洞的独立轮廓+凹陷也视为洞

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# a) Concrete behavior and trace:
#    - Keep the original stage: separated blue outer objects on yellow background with red 8-neighborhood
#      outer outlines in the output.
#    - The fused variant introduces three visually distinct "hole-like" situations inside a blue shell:
#      1) a true enclosed background hole,
#      2) an enclosed pocket filled by a green nested object,
#      3) a deep narrow indentation that opens to the shell bbox edge.
#    - Local trace:
#      old task: blue ring + yellow 3x3 hole -> teal shell, magenta hole, red outer outline.
#      fused task adds two changes:
#      * the same 3x3 pocket may instead be green in the input, but it is still treated as a hole-like
#        region in the output;
#      * every hole-like region gets its own orange inner rim in addition to the red outer outline.
#      So a nested object becomes teal shell + orange-rimmed magenta core, and a deep notch becomes a
#      filled orange/magenta cavity inside a teal object with the outer red outline computed after filling.
# b) Complexity calibration:
#    - Each sample always includes one nested object, one classical enclosed hole, and one deep notch.
#      This prevents the fusion from degenerating into a single-branch task and gives repeated support
#      for each subrule across examples.
#    - Hole and nested-pocket rectangles are restricted to 3x3 / 3x4 / 4x4 so the orange inner rim and
#      magenta interior are both usually visible.
#    - Deep notches stay width 1-2 and depth 3-5, while an optional shallow/wide notch negative example
#      shows that not every concavity is promoted to a hole.
# c) Implicit-assumption audit for main():
#    - Iteration must no longer cover every non-background component, because green nested objects are not
#      top-level objects. `main()` therefore iterates only blue shell components.
#    - The original "_interior holes are bg-only" assumption is broken by nested objects. The new solver
#      treats every enclosed pocket of the blue shell as hole-like regardless of whether the pocket is
#      yellow or green in the input.
#    - The original "holes do not touch the bbox edge" assumption is also insufficient because deep
#      indentations intentionally do touch the bbox edge. A second detector scans bbox-edge runs for
#      width<=2, depth>=3 rectangular notches with full side walls and a solid back wall.
#    - The outer red outline must be computed from the shell after all hole-like regions are filled into
#      the solid mask; otherwise red would leak into the interior pocket or down the indentation throat.
#    - The orange inner border must be drawn on the special-region cells themselves, before the red outer
#      pass, so the two outline systems never compete for the same cells.
# d) Placement legality:
#    - Generator placement uses a conservative occupied mask: shell union pocket/notch. This is stricter
#      than the raw input occupancy, but it guarantees that no later component can intrude into a hole or
#      notch that will be filled by the transform, and it preserves the non-overlap of outer red outlines.
#    - Objects are still kept Chebyshev distance >= 3 apart and at least one cell from the canvas edge.
# e) Visual observability:
#    - Humans can distinguish the three cases directly in the input:
#      * true hole: empty yellow cavity inside blue,
#      * nested object: green cavity filler inside blue,
#      * deep notch: blue object with a thin, deep mouth to the outside.
#    - In the output, all three receive the same teal body + orange/magenta internal treatment, making
#      the abstraction "these are all holes" inferable from 2-4 examples.
# f) Visual cleanliness:
#    - Only three new semantic colors appear in output: teal shell, magenta hole fill, orange hole rim.
#      Red is reserved for outside background cells only, so the nested two-boundary structure remains legible.
#    - Each special object carries at most one special region, avoiding clutter from overlapping markers.
# g) Change inferability:
#    - The color change is structurally anchored instead of arbitrary: only blue shells with a visible
#      enclosed pocket or a qualifying deep notch become teal; the special region itself becomes
#      magenta-with-orange-rim. A plain solid or shallow/wide notch remains blue with only a red exterior.
# h) Verification metric used during development:
#    - Every generated sample must contain at least one nested pocket, one enclosed background hole, and
#      one deep notch; outputs must contain orange and magenta; and the fused solver output must differ
#      from the original solver on these generated inputs.
# i) New solver assumptions enforced by the generator:
#    - Nested pockets are fully filled by a single green rectangular inner object.
#    - Deep notches are axis-aligned rectangles with mouth width 1-2 and depth >= 3.
#    - Negative concavities, when present, are shallow or wide enough to avoid accidental promotion.


BG_COLOR = 4
FG_COLOR = 1
NESTED_COLOR = 3
OUTLINE_COLOR = 2
HOLE_FILL_COLOR = 6
HOLE_INNER_OUTLINE_COLOR = 7
SPECIAL_BODY_COLOR = 8


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _edge_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        if not bool(line[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(line[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _interior_pockets_in_sub(sub: np.ndarray) -> list[np.ndarray]:
    pockets: list[np.ndarray] = []
    h, w = sub.shape
    for pocket in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (pocket[:, 0] == 0)
            | (pocket[:, 1] == 0)
            | (pocket[:, 0] == h - 1)
            | (pocket[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            pockets.append(pocket)
    return pockets


def _deep_notch_regions_in_sub(sub: np.ndarray) -> list[np.ndarray]:
    h, w = sub.shape
    regions: list[np.ndarray] = []

    for c0, c1 in _edge_runs(~sub[0]):
        width = c1 - c0 + 1
        if width > 2 or c0 == 0 or c1 == w - 1:
            continue
        depth = 0
        while depth < h:
            if sub[depth, c0 : c1 + 1].any():
                break
            if not bool(sub[depth, c0 - 1]) or not bool(sub[depth, c1 + 1]):
                break
            depth += 1
        if depth >= 3 and depth < h and sub[depth, c0 : c1 + 1].all():
            regions.append(
                np.array([(r, c) for r in range(depth) for c in range(c0, c1 + 1)], dtype=int)
            )

    for c0, c1 in _edge_runs(~sub[h - 1]):
        width = c1 - c0 + 1
        if width > 2 or c0 == 0 or c1 == w - 1:
            continue
        depth = 0
        while depth < h:
            r = h - 1 - depth
            if sub[r, c0 : c1 + 1].any():
                break
            if not bool(sub[r, c0 - 1]) or not bool(sub[r, c1 + 1]):
                break
            depth += 1
        close_r = h - 1 - depth
        if depth >= 3 and close_r >= 0 and sub[close_r, c0 : c1 + 1].all():
            regions.append(
                np.array(
                    [(r, c) for r in range(h - depth, h) for c in range(c0, c1 + 1)],
                    dtype=int,
                )
            )

    for r0, r1 in _edge_runs(~sub[:, 0]):
        width = r1 - r0 + 1
        if width > 2 or r0 == 0 or r1 == h - 1:
            continue
        depth = 0
        while depth < w:
            if sub[r0 : r1 + 1, depth].any():
                break
            if not bool(sub[r0 - 1, depth]) or not bool(sub[r1 + 1, depth]):
                break
            depth += 1
        if depth >= 3 and depth < w and sub[r0 : r1 + 1, depth].all():
            regions.append(
                np.array([(r, c) for r in range(r0, r1 + 1) for c in range(depth)], dtype=int)
            )

    for r0, r1 in _edge_runs(~sub[:, w - 1]):
        width = r1 - r0 + 1
        if width > 2 or r0 == 0 or r1 == h - 1:
            continue
        depth = 0
        while depth < w:
            c = w - 1 - depth
            if sub[r0 : r1 + 1, c].any():
                break
            if not bool(sub[r0 - 1, c]) or not bool(sub[r1 + 1, c]):
                break
            depth += 1
        close_c = w - 1 - depth
        if depth >= 3 and close_c >= 0 and sub[r0 : r1 + 1, close_c].all():
            regions.append(
                np.array(
                    [(r, c) for r in range(r0, r1 + 1) for c in range(w - depth, w)],
                    dtype=int,
                )
            )

    return regions


def _shift_regions(regions: list[np.ndarray], r0: int, c0: int) -> list[np.ndarray]:
    shifted_regions: list[np.ndarray] = []
    for region in regions:
        shifted = region.copy()
        shifted[:, 0] += int(r0)
        shifted[:, 1] += int(c0)
        shifted_regions.append(shifted)
    return shifted_regions


def _interior_pockets(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return _shift_regions(_interior_pockets_in_sub(sub), r0, c0)


def _deep_notch_regions(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return _shift_regions(_deep_notch_regions_in_sub(sub), r0, c0)


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


def _inner_perimeter(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = rows + dr
        cc = cols + dc
        out_of_bounds = (rr < 0) | (rr >= h) | (cc < 0) | (cc >= w)
        if np.any(out_of_bounds):
            border[rows[out_of_bounds], cols[out_of_bounds]] = True
        ok = ~out_of_bounds
        if np.any(ok):
            exposed = ~mask[rr[ok], cc[ok]]
            border[rows[ok][exposed], cols[ok][exposed]] = True
    return border


def _feature_counts(mask: np.ndarray) -> tuple[int, int]:
    return len(_interior_pockets_in_sub(mask)), len(_deep_notch_regions_in_sub(mask))


def _has_special_region(mask: np.ndarray) -> bool:
    enclosed, deep = _feature_counts(mask)
    return enclosed > 0 or deep > 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor shell objects with hole-like regions and add both outer and inner outlines.

    Input structure:
    - A mostly uniform background contains several separated blue 4-connected outer shells.
    - Some blue shells enclose a rectangular pocket that is either empty background or fully filled
      by a green nested object, and some shells instead expose a deep narrow notch from one bbox edge.

    Rule:
    - Every blue outer shell receives a one-cell 8-neighborhood red outline on background cells only.
    - If a shell has any enclosed pocket or qualifying deep notch, recolor the blue shell to teal.
    - Recolor each such hole-like region to magenta, then recolor that region's own 4-neighbor inner
      rim to orange, and compute the red outer outline on the shell after those regions are treated as filled.

    Key insight:
    - The transform is driven by the geometry of the blue shell rather than the original fill of the
      cavity: an empty enclosed hole, a green nested core, and a deep narrow indentation are all
      normalized into the same "filled cavity + independent inner border" representation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[FG_COLOR]):
        coords = comp["coords"]
        shell_mask = _coords_mask(g.shape, coords)
        pockets = _interior_pockets(shell_mask, coords)
        deep_notches = _deep_notch_regions(shell_mask, coords)
        special_regions = pockets + deep_notches

        solid = shell_mask.copy()
        if special_regions:
            out[shell_mask] = SPECIAL_BODY_COLOR
            for region in special_regions:
                region_mask = _coords_mask(g.shape, region)
                out[region[:, 0], region[:, 1]] = HOLE_FILL_COLOR
                out[_inner_perimeter(region_mask)] = HOLE_INNER_OUTLINE_COLOR
                solid |= region_mask

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


# --------------- generator (c) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

# Component mix: always show the three promoted region types.
COMP_COUNT_RANGE = (3, 4)
EXTRA_COMPONENT_SHALLOW_PROB = 0.75

# Polyomino construction for clean solids
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

# Small solid shapes
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Shell-with-pocket construction
POCKET_VALID_AREAS = [9, 12, 16]
POCKET_DIM_MAX = 4
SHELL_PAD_RANGE = (1, 3)
SHELL_MAX_NOTCHES = 2
SHELL_EXTENSION_PROB = 0.5
SHELL_EXT_H_RANGE = (1, 4)
SHELL_EXT_SIDE_W_RANGE = (1, 3)
SHELL_POCKET_SEARCH_RETRIES = 200
SHELL_GEN_RETRIES = 50
SHELL_MIN_AREA = 10

# Boundary-notch construction
NOTCH_BASE_H_RANGE = (6, 10)
NOTCH_BASE_W_RANGE = (6, 10)
DEEP_NOTCH_DEPTH_RANGE = (3, 5)
SHALLOW_NOTCH_DEPTH_RANGE = (1, 2)
NOTCH_GEN_RETRIES = 40
NOTCH_COMPONENT_MIN_AREA = 12

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


def _chebyshev_conflict(
    rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, placed_mask: np.ndarray
) -> bool:
    rows_b, cols_b = np.where(placed_mask)
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
    if _has_special_region(mask):
        return None
    return mask


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int) -> np.ndarray:
    if random.random() >= SHELL_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*SHELL_EXT_H_RANGE)

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
        ext_w_actual = random.randint(*SHELL_EXT_SIDE_W_RANGE)
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
    for _ in range(SHELL_POCKET_SEARCH_RETRIES):
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
        enclosed, deep = _feature_counts(cropped)
        if enclosed == 1 and deep == 0:
            return cropped
    return None


def _make_shell_with_pocket(target_pocket_area: int) -> np.ndarray | None:
    dims = _rect_dims(target_pocket_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*SHELL_PAD_RANGE)
    pad_bot = random.randint(*SHELL_PAD_RANGE)
    pad_left = random.randint(*SHELL_PAD_RANGE)
    pad_right = random.randint(*SHELL_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, SHELL_MAX_NOTCHES)
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


def _carve_edge_notch(mask: np.ndarray, *, side: str, width: int, depth: int, offset: int) -> np.ndarray:
    carved = mask.copy()
    if side == "top":
        carved[0:depth, offset : offset + width] = False
    elif side == "bottom":
        carved[-depth:, offset : offset + width] = False
    elif side == "left":
        carved[offset : offset + width, 0:depth] = False
    else:
        carved[offset : offset + width, -depth:] = False
    return carved


def _build_pocket_component(kind: str) -> dict[str, np.ndarray] | None:
    for _ in range(SHELL_GEN_RETRIES):
        outer = _make_shell_with_pocket(random.choice(POCKET_VALID_AREAS))
        if outer is None or int(outer.sum()) < SHELL_MIN_AREA:
            continue
        pockets = _interior_pockets_in_sub(outer)
        if len(pockets) != 1:
            continue
        pocket_mask = _coords_mask(outer.shape, pockets[0])
        occupied = outer | pocket_mask
        if not is_contiguous(occupied.astype(int), background=0, connectivity=4):
            continue
        inner = pocket_mask if kind == "nested" else np.zeros_like(outer, dtype=bool)
        return {"outer": outer, "inner": inner, "occupied": occupied}
    return None


def _build_deep_notch_component() -> dict[str, np.ndarray] | None:
    for _ in range(NOTCH_GEN_RETRIES):
        base_h = random.randint(*NOTCH_BASE_H_RANGE)
        base_w = random.randint(*NOTCH_BASE_W_RANGE)
        outer = np.ones((base_h, base_w), dtype=bool)
        side = random.choice(["top", "bottom", "left", "right"])

        if side in ("top", "bottom"):
            width = random.randint(1, 2)
            max_depth = min(DEEP_NOTCH_DEPTH_RANGE[1], base_h - 2)
            if max_depth < DEEP_NOTCH_DEPTH_RANGE[0] or base_w - width - 2 < 0:
                continue
            depth = random.randint(DEEP_NOTCH_DEPTH_RANGE[0], max_depth)
            offset = random.randint(1, base_w - width - 1)
        else:
            width = random.randint(1, 2)
            max_depth = min(DEEP_NOTCH_DEPTH_RANGE[1], base_w - 2)
            if max_depth < DEEP_NOTCH_DEPTH_RANGE[0] or base_h - width - 2 < 0:
                continue
            depth = random.randint(DEEP_NOTCH_DEPTH_RANGE[0], max_depth)
            offset = random.randint(1, base_h - width - 1)

        outer = _carve_edge_notch(outer, side=side, width=width, depth=depth, offset=offset)
        if not is_contiguous(outer.astype(int), background=0, connectivity=4):
            continue
        enclosed, deep = _feature_counts(outer)
        if enclosed != 0 or deep != 1 or int(outer.sum()) < NOTCH_COMPONENT_MIN_AREA:
            continue
        notch_mask = _coords_mask(outer.shape, _deep_notch_regions_in_sub(outer)[0])
        occupied = outer | notch_mask
        return {"outer": outer, "inner": np.zeros_like(outer, dtype=bool), "occupied": occupied}
    return None


def _build_shallow_notch_component() -> dict[str, np.ndarray] | None:
    for _ in range(NOTCH_GEN_RETRIES):
        base_h = random.randint(*NOTCH_BASE_H_RANGE)
        base_w = random.randint(*NOTCH_BASE_W_RANGE)
        outer = np.ones((base_h, base_w), dtype=bool)
        side = random.choice(["top", "bottom", "left", "right"])
        use_shallow_depth = random.random() < 0.5

        if side in ("top", "bottom"):
            if use_shallow_depth:
                width = random.randint(1, 2)
                max_depth = min(SHALLOW_NOTCH_DEPTH_RANGE[1], base_h - 2)
                if max_depth < SHALLOW_NOTCH_DEPTH_RANGE[0] or base_w - width - 2 < 0:
                    continue
                depth = random.randint(SHALLOW_NOTCH_DEPTH_RANGE[0], max_depth)
            else:
                width = 3
                max_depth = min(DEEP_NOTCH_DEPTH_RANGE[1], base_h - 2)
                if base_w - width - 2 < 0 or max_depth < DEEP_NOTCH_DEPTH_RANGE[0]:
                    continue
                depth = random.randint(DEEP_NOTCH_DEPTH_RANGE[0], max_depth)
            offset = random.randint(1, base_w - width - 1)
        else:
            if use_shallow_depth:
                width = random.randint(1, 2)
                max_depth = min(SHALLOW_NOTCH_DEPTH_RANGE[1], base_w - 2)
                if max_depth < SHALLOW_NOTCH_DEPTH_RANGE[0] or base_h - width - 2 < 0:
                    continue
                depth = random.randint(SHALLOW_NOTCH_DEPTH_RANGE[0], max_depth)
            else:
                width = 3
                max_depth = min(DEEP_NOTCH_DEPTH_RANGE[1], base_w - 2)
                if base_h - width - 2 < 0 or max_depth < DEEP_NOTCH_DEPTH_RANGE[0]:
                    continue
                depth = random.randint(DEEP_NOTCH_DEPTH_RANGE[0], max_depth)
            offset = random.randint(1, base_h - width - 1)

        outer = _carve_edge_notch(outer, side=side, width=width, depth=depth, offset=offset)
        if not is_contiguous(outer.astype(int), background=0, connectivity=4):
            continue
        enclosed, deep = _feature_counts(outer)
        if enclosed == 0 and deep == 0 and int(outer.sum()) >= NOTCH_COMPONENT_MIN_AREA:
            return {
                "outer": outer,
                "inner": np.zeros_like(outer, dtype=bool),
                "occupied": outer.copy(),
            }
    return None


def _build_clean_solid() -> dict[str, np.ndarray] | None:
    if random.random() < SMALL_SHAPE_PROB:
        outer = _make_small_shape()
        if _has_special_region(outer):
            return None
        return {"outer": outer, "inner": np.zeros_like(outer, dtype=bool), "occupied": outer.copy()}

    for _ in range(POLY_GEN_RETRIES):
        outer = _make_polyomino()
        if outer is not None:
            return {"outer": outer, "inner": np.zeros_like(outer, dtype=bool), "occupied": outer.copy()}
    return None


def _build_component(kind: str) -> dict[str, np.ndarray] | None:
    if kind == "nested":
        return _build_pocket_component("nested")
    if kind == "hole":
        return _build_pocket_component("hole")
    if kind == "deep":
        return _build_deep_notch_component()
    if kind == "shallow":
        return _build_shallow_notch_component()
    return _build_clean_solid()


def _planned_kinds(num_components: int) -> list[str]:
    kinds = ["nested", "hole", "deep"]
    if num_components == 4:
        kinds.append("shallow" if random.random() < EXTRA_COMPONENT_SHALLOW_PROB else "solid")
    random.shuffle(kinds)
    return kinds


def _input_feature_signature(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    nested = 0
    holes = 0
    deep = 0

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[FG_COLOR]):
        coords = comp["coords"]
        shell = _coords_mask(g.shape, coords)
        for pocket in _interior_pockets(shell, coords):
            vals = g[pocket[:, 0], pocket[:, 1]]
            if np.all(vals == bg):
                holes += 1
            elif np.all(vals == NESTED_COLOR):
                nested += 1
        deep += len(_deep_notch_regions(shell, coords))

    return nested, holes, deep


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    num_components = random.randint(*COMP_COUNT_RANGE)

    kinds = _planned_kinds(num_components)
    components: list[dict[str, np.ndarray]] = []
    for kind in kinds:
        comp = _build_component(kind)
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
        rows_a, cols_a = np.where(occ)

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

            rows_here, cols_here = rows_a + r, cols_a + c
            conflict = False
            for pr, pc, prev in placed:
                prev_occ = prev["occupied"]
                row_gap = max(0, max(r - (pr + prev_occ.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + prev_occ.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_here, cols_here, pr, pc, prev_occ):
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
        grid[inner_rows + r, inner_cols + c] = NESTED_COLOR

    nested_count, hole_count, deep_count = _input_feature_signature(grid)
    if nested_count < 1 or hole_count < 1 or deep_count < 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out == HOLE_INNER_OUTLINE_COLOR):
        return None
    if not np.any(out == HOLE_FILL_COLOR):
        return None
    if not np.any(out == SPECIAL_BODY_COLOR):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
