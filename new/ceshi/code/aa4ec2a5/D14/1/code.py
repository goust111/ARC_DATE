# Variant of aa4ec2a5
# Dimension: D14 — 对象的形态复杂度 (meta)
# Change: 蓝色4连通对象为任意形状，部分对象内部有封闭洞，对象间互不接触... → 对象为矩形框架（空心矩形）

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg


# Design analysis for the D14 variant
# -----------------------------------
# variant_spec:
#   grid:
#     rows: 15-28
#     cols: 15-27
#     background: yellow (4)
#   objects:
#     count: 3-4 blue (1) 4-connected components
#     shape: every component is a single axis-aligned hollow rectangle with uniform thickness
#     hole: exactly one rectangular interior cavity per component
#     thickness_classes:
#       narrow: width == 1, stays blue
#       wide: width >= 2, recolored teal
#   transform:
#     - fill every interior cavity with magenta (6)
#     - treat the filled cavity as part of the solid region when drawing the red (2) 8-neighborhood outline
#     - recolor only wide frames to teal (8); narrow frames remain blue (1)
#
# behavior_trace:
#   original logic on a hollow object:
#     "has any interior hole?" -> yes -> recolor full object teal, fill hole magenta, outline the hole-filled mask
#   variant logic on a 7x9 frame:
#     if hole bbox implies thickness 1 -> keep frame blue, fill hole magenta, outline outer rectangle
#     if hole bbox implies thickness 2 -> recolor frame teal, fill hole magenta, outline outer rectangle
#   key difference:
#     all objects now have holes, so hole existence no longer separates classes; only geometric frame width does.
#
# complexity_calibration:
#   - Every sample contains both narrow and wide frames, so the blue/teal split is observable in 2-3 examples.
#   - Width takes only a few values (1, 2, 3), which keeps the rule inferable while still being geometric.
#   - Objects remain separated by Chebyshev distance >= 3 so red outlines do not merge visually.
#
# implicit_assumption_audit:
#   - Original line `if holes: out[mask] = 8` depended on "hole existence implies teal".
#     That becomes wrong here because all frames have holes but narrow ones must stay blue.
#   - Original hole-finding helper is still valid, but solver now also assumes:
#       1. each object has exactly one enclosed hole,
#       2. the outer bbox minus the hole bbox forms a uniform rectangular frame.
#     The generator guarantees both.
#   - Outline drawing still assumes holes are merged into the solid mask before outlining.
#
# placement_legality_and_visuals:
#   - Generator uses only axis-aligned frames, places largest first, and rejects Chebyshev conflicts.
#   - Border clearance stays >= 1, so the red outline may touch the canvas edge but objects themselves do not.
#   - Because each object has a simple frame + cavity + halo, outputs stay visually clean and easy to parse.


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
    Add red outlines around hollow rectangular frames, and classify them by frame thickness.

    Input structure:
    - A mostly uniform background contains several blue 4-connected components.
    - Every component is a hollow axis-aligned rectangular frame with exactly one enclosed cavity.

    Rule:
    - Fill each enclosed cavity with magenta (6), and include that cavity when computing the outline.
    - Draw a one-cell 8-neighborhood outline in red (2) around every resulting solid rectangle.
    - Recolor only wide frames (thickness >= 2) to teal (8); narrow one-pixel frames stay blue (1).

    Key insight: The presence of a hole is no longer the deciding feature because every object is hollow;
    the solver must measure the rectangular frame's width to decide whether the frame stays blue or turns teal.
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
            out[hole[:, 0], hole[:, 1]] = 6
            solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
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


def _choose_hole_dims(thickness: int, H: int, W: int, num_components: int) -> tuple[int, int] | None:
    small_grid = min(H, W) < 20 or num_components == 4
    lo, hi = SMALL_HOLE_DIM_RANGE if small_grid else LARGE_HOLE_DIM_RANGE
    max_hole_h = min(hi, H - 2 * BORDER_MIN - 2 * thickness)
    max_hole_w = min(hi, W - 2 * BORDER_MIN - 2 * thickness)
    if max_hole_h < lo or max_hole_w < lo:
        return None
    return random.randint(lo, max_hole_h), random.randint(lo, max_hole_w)


def _chebyshev_conflict(rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray) -> bool:
    rows_b, cols_b = np.where(pm)
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


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    max_components = COMP_COUNT_RANGE[1] if min(H, W) >= 20 else 3
    num_components = random.randint(COMP_COUNT_RANGE[0], max_components)
    num_narrow = random.randint(NARROW_COUNT_RANGE[0], min(NARROW_COUNT_RANGE[1], num_components - 1))

    wide_choices = [t for t in WIDE_THICKNESS_CHOICES if H >= 2 * t + 4 and W >= 2 * t + 4]
    if not wide_choices:
        return None

    thicknesses = [NARROW_THICKNESS] * num_narrow
    thicknesses.extend(random.choice(wide_choices) for _ in range(num_components - num_narrow))
    random.shuffle(thicknesses)

    components: list[np.ndarray] = []
    for thickness in thicknesses:
        dims = _choose_hole_dims(thickness, H, W, num_components)
        if dims is None:
            return None
        hole_h, hole_w = dims
        components.append(_make_frame(thickness, hole_h, hole_w))

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            max_r = H - mh - BORDER_MIN
            max_c = W - mw - BORDER_MIN
            if max_r < BORDER_MIN or max_c < BORDER_MIN:
                return None

            r = random.randint(BORDER_MIN, max_r)
            c = random.randint(BORDER_MIN, max_c)
            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

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
        grid[rows + r, cols + c] = FG_COLOR

    try:
        thickness_list = sorted(_component_thicknesses(grid))
        if not (any(t == 1 for t in thickness_list) and any(t >= 2 for t in thickness_list)):
            return None
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
