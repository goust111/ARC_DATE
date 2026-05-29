# Combo variant of aa4ec2a5 (index 14)
# combo_spec: D5/1+D14/1
# Name: 矩形框架对象+洞填充为红色

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg


# Design analysis for combo D5/1 + D14/1
# ----------------------------------------
# concretization_and_trace:
#   fused_input:
#     - yellow background 4
#     - 3-4 blue 4-connected components
#     - every component is a hollow axis-aligned rectangular frame with exactly one
#       enclosed rectangular hole
#     - the only latent class feature is frame thickness:
#         * thickness 1 -> object stays blue
#         * thickness >= 2 -> object becomes teal
#   original_task_trace_on_one_hollow_shape:
#     - detect any enclosed hole
#     - because hole exists, recolor full object teal
#     - recolor hole magenta
#     - merge hole into silhouette and draw red outer halo
#   fused_trace_on_two frames:
#     - example A: outer 6x8, hole 4x6, thickness 1
#       * old task: hole existence would force teal + magenta hole
#       * fused task: measure thickness == 1, keep frame blue, fill hole red, then
#         outline the hole-filled rectangle in red
#     - example B: outer 8x10, hole 4x6, thickness 2
#       * old task: teal + magenta hole
#       * fused task: measure thickness == 2, recolor frame teal, fill hole red,
#         then outline the hole-filled rectangle in red
#   visible_difference:
#     - all objects are hollow, so "has a hole" stops being discriminative
#     - red now has two geometric roles at once:
#         * outside the frame it is the 8-neighborhood halo
#         * inside the frame it is the filled cavity
#     - the class cue is no longer topology but frame width
#
# complexity_calibration:
#   - Thickness is intentionally limited to a few values (1, 2, 3) so the blue/teal
#     split is learnable from a handful of examples.
#   - Every generated grid contains at least one narrow frame and at least one wide
#     frame; otherwise the thickness-conditioned recolor rule would be underdetermined.
#   - Hole dimensions are kept at least 2x2 so the inner red fill is visually obvious
#     and cannot be mistaken for a single noisy pixel.
#   - All objects remain rectangles with one hole each, which removes excess shape
#     freedom and makes the fused ambiguity focused on width + red-role disambiguation.
#
# implicit_assumption_review_main:
#   - `_interior_holes(...)` still works because hole detection is purely geometric.
#   - The original assumption "if holes: recolor object teal" is now false because
#     every object has a hole and only wide frames should turn teal.
#   - The original assumption "hole color is a dedicated magenta marker" is also false;
#     hole fill must change to red so it visually matches the outer outline.
#   - `_frame_thickness(...)` introduces a new assumption: each component must be a
#     uniform rectangular frame with exactly one enclosed hole. The generator enforces
#     that by constructing components directly from outer rectangle minus inner rectangle.
#   - `solid[hole] = True` remains essential; otherwise the outline would hug both the
#     outer and inner boundaries instead of only the exterior rectangle.
#   - `out[outline & (out == bg)] = 2` still matters because the red-filled hole should
#     not be treated as background and re-outlined internally.
#
# placement_legality:
#   - Components are generated as explicit boolean masks and placed largest-first.
#   - Placement keeps border clearance >= 1 and pairwise Chebyshev distance >= 3
#     between component cells, so red halos cannot overlap and objects never leave the
#     canvas.
#   - The generator rejects layouts that cannot fit both narrow and wide frames under
#     these spacing rules, avoiding degenerate overpacked samples.
#
# observability_and_cleanliness:
#   - Humans can compare a thin blue frame and a thick teal frame that both contain red
#     interiors; that repeated contrast reveals that thickness, not hole presence,
#     controls the body recolor.
#   - Using the same red inside and outside creates the intended ambiguity, but the
#     geometry stays clean because interior red is fully surrounded by the frame while
#     exterior red is fully outside the bbox.
#   - No hidden ordering, numeric-color ranking, or non-visual metadata is used.
#
# inferability_of_color_change:
#   - Only four colors appear in outputs: yellow background, blue narrow frames, teal
#     wide frames, red for both hole fill and outer halo.
#   - The red reuse is still inferable because its two occurrences are structurally
#     separated: enclosed rectangular patch versus external one-pixel halo.
#   - Because every sample mixes thickness classes, the blue/teal decision can be
#     inferred from repeated geometric evidence rather than a one-off coincidence.
#
# post_change_assumption_audit:
#   - `main()` loops over monochromatic connected components and assumes each component's
#     bbox minus its unique hole defines a uniform frame. `_frame_thickness` checks that.
#   - `generate_input()` guarantees the assumption by only emitting masks produced by
#     `_make_frame(...)`, then validating all discovered components recover thicknesses.
#   - `main()` also assumes component outlines do not need conflict resolution beyond
#     "paint background only"; generator spacing preserves that invariant.


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


def _frame_thickness(mask: np.ndarray, coords: np.ndarray, holes: list[np.ndarray]) -> int | None:
    if len(holes) != 1:
        return None

    r0, c0, r1, c1 = bbox_coords(coords)
    hr0, hc0, hr1, hc1 = bbox_coords(holes[0])
    top = hr0 - r0
    bottom = r1 - hr1
    left = hc0 - c0
    right = c1 - hc1
    widths = [int(top), int(bottom), int(left), int(right)]
    if min(widths) < 1 or len(set(widths)) != 1:
        return None

    thickness = widths[0]
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.ones_like(sub, dtype=bool)
    expected[thickness : sub.shape[0] - thickness, thickness : sub.shape[1] - thickness] = False
    if not np.array_equal(sub, expected):
        return None
    return thickness


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red halos around hollow rectangular frames and fill their cavities with the same red.

    Input structure:
    - A uniform yellow background contains several blue 4-connected components.
    - Every component is an axis-aligned hollow rectangular frame with exactly one enclosed hole.

    Rule:
    - Measure each frame's uniform thickness from its outer rectangle to its hole.
    - If the thickness is 1, keep the frame blue; if the thickness is 2 or more, recolor the frame teal (8).
    - Fill the enclosed hole with red (2), then treat that filled hole as part of the solid region and draw a one-cell 8-neighborhood red outline on background cells only.

    Key insight:
    - All objects have holes, so hole presence no longer determines recoloring. The solver must infer class from geometric frame width, while the same red color simultaneously marks interior hole fill and exterior outline.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        thickness = _frame_thickness(mask, coords, holes)

        solid = mask.copy()
        if thickness is not None and thickness >= 2:
            out[mask] = 8

        for hole in holes:
            out[hole[:, 0], hole[:, 1]] = 2
            solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (combo) ---------------

GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
NARROW_COUNT_RANGE = (1, 2)

NARROW_THICKNESS = 1
WIDE_THICKNESS_CHOICES = (2, 3)
SMALL_HOLE_DIM_RANGE = (2, 3)
LARGE_HOLE_DIM_RANGE = (2, 5)

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 400
GEN_OUTER_RETRIES = 400


def _make_frame(thickness: int, hole_h: int, hole_w: int) -> np.ndarray:
    outer_h = hole_h + 2 * thickness
    outer_w = hole_w + 2 * thickness
    mask = np.ones((outer_h, outer_w), dtype=bool)
    mask[thickness:-thickness, thickness:-thickness] = False
    return mask


def _choose_hole_dims(
    thickness: int, grid_h: int, grid_w: int, num_components: int
) -> tuple[int, int] | None:
    small_grid = min(grid_h, grid_w) < 20 or num_components == 4
    lo, hi = SMALL_HOLE_DIM_RANGE if small_grid else LARGE_HOLE_DIM_RANGE
    max_hole_h = min(hi, grid_h - 2 * BORDER_MIN - 2 * thickness)
    max_hole_w = min(hi, grid_w - 2 * BORDER_MIN - 2 * thickness)
    if max_hole_h < lo or max_hole_w < lo:
        return None
    return random.randint(lo, max_hole_h), random.randint(lo, max_hole_w)


def _chebyshev_conflict(
    rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, placed_mask: np.ndarray
) -> bool:
    rows_b, cols_b = np.where(placed_mask)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _component_thicknesses(grid: np.ndarray) -> list[int]:
    bg = get_bg(grid)
    thicknesses: list[int] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        thickness = _frame_thickness(mask, coords, holes)
        if thickness is None:
            raise ValueError("generated component is not a uniform rectangular frame")
        thicknesses.append(thickness)
    return thicknesses


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = get_bg(inp)
    metrics = {
        "component_count": 0,
        "narrow_count": 0,
        "wide_count": 0,
        "red_hole_cells": 0,
        "teal_body_cells": 0,
        "blue_body_cells": 0,
    }

    for comp in find_components(inp, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(inp.shape, coords)
        holes = _interior_holes(mask, coords)
        thickness = _frame_thickness(mask, coords, holes)
        if thickness is None:
            continue

        metrics["component_count"] += 1
        if thickness == 1:
            metrics["narrow_count"] += 1
        else:
            metrics["wide_count"] += 1

        metrics["blue_body_cells"] += int(np.sum(out[mask] == 1))
        metrics["teal_body_cells"] += int(np.sum(out[mask] == 8))
        for hole in holes:
            metrics["red_hole_cells"] += int(np.sum(out[hole[:, 0], hole[:, 1]] == 2))

    return metrics


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    max_components = COMP_COUNT_RANGE[1] if min(grid_h, grid_w) >= 20 else 3
    num_components = random.randint(COMP_COUNT_RANGE[0], max_components)
    num_narrow = random.randint(NARROW_COUNT_RANGE[0], min(NARROW_COUNT_RANGE[1], num_components - 1))

    wide_choices = [
        thickness
        for thickness in WIDE_THICKNESS_CHOICES
        if grid_h >= 2 * thickness + 4 and grid_w >= 2 * thickness + 4
    ]
    if not wide_choices:
        return None

    thicknesses = [NARROW_THICKNESS] * num_narrow
    thicknesses.extend(random.choice(wide_choices) for _ in range(num_components - num_narrow))
    random.shuffle(thicknesses)

    components: list[np.ndarray] = []
    for thickness in thicknesses:
        dims = _choose_hole_dims(thickness, grid_h, grid_w, num_components)
        if dims is None:
            return None
        hole_h, hole_w = dims
        components.append(_make_frame(thickness, hole_h, hole_w))

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        max_r = grid_h - mh - BORDER_MIN
        max_c = grid_w - mw - BORDER_MIN
        if max_r < BORDER_MIN or max_c < BORDER_MIN:
            return None

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max_r)
            c = random.randint(BORDER_MIN, max_c)
            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, placed_mask in placed:
                row_gap = max(0, max(r - (pr + placed_mask.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + placed_mask.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, placed_mask):
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
        thickness_list = _component_thicknesses(grid)
        if not any(t == 1 for t in thickness_list):
            return None
        if not any(t >= 2 for t in thickness_list):
            return None

        out = main(grid)
        metrics = _fusion_metrics(grid, out)
        if metrics["component_count"] != num_components:
            return None
        if metrics["narrow_count"] < 1 or metrics["wide_count"] < 1:
            return None
        if metrics["red_hole_cells"] <= 0:
            return None
        if metrics["blue_body_cells"] <= 0 or metrics["teal_body_cells"] <= 0:
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
