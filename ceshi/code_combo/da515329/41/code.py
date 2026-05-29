# Combo variant of da515329 (index 41)
# combo_spec: D6/1+D18/0
# Name: 固定段长 + 同心矩形框

import math
import random

import numpy as np

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the D6/1 + D18/0 combo
# -------------------------------------------
# spec_variant:
#   preserved_input:
#     grid: even-sized square black canvas with one teal center-hole cross
#     cross: four equal axis-aligned arms, length 1..4, color 8
#   fused_output:
#     D6/1 contribution:
#       replace the original connected spiral corridor with closed rectangular
#       loops derived from the arm-tip bounding box
#     D18/0 contribution:
#       keep every loop at the same side length instead of expanding outward
#     nondegenerate fusion:
#       if equal-sized loops stayed centered on the same cell, every later loop
#       would redraw the first frame and the rule would collapse to one box
#       only; to keep the "many frames" signal visible, each extra layer copies
#       the same seed frame onto the four diagonal centers (cr +/- 2k, cc +/- 2k)
#       for k = 1, 2, ...
#     spacing:
#       the diagonal stride is 2 cells, matching the source task's 1-cell wall /
#       1-cell gap rhythm; frames may overlap, but their edge lengths never grow
#   concrete_trace:
#     example_arm_2:
#       input tips lie on a 5x5 box around the center hole
#       original solver would launch four right-turning turtles and progressively
#       lengthen segments to build one outward spiral maze
#       D6-only would draw 5x5, then 9x9, then 13x13 concentric frames
#       fused rule keeps the 5x5 frame size fixed and draws copies centered at
#       (cr, cc), then (cr +/- 2, cc +/- 2), then (cr +/- 4, cc +/- 4), etc.
#       the visible difference is "same-sized repeated boxes" rather than
#       "larger and larger boxes"
#   complexity_calibration:
#       arm length still controls the box size, and border margin controls how
#       many diagonal copies fit; this keeps samples varied without introducing
#       extra colors or brittle case splits
#   implicit_assumption_audit:
#     removed_assumptions:
#       all original turtle state, turn priority, partial 2-step moves, parity
#       cleanup, and border nub fixes assumed one self-avoiding path whose local
#       geometry depended on growing segment lengths; none of that survives once
#       the path becomes a family of fixed-size disconnected loops
#     new_solver_assumptions:
#       there is exactly one teal center-hole cross
#       the four arms are equal and axis-aligned
#       the intended pattern is described entirely by the cross center, arm
#       length, and which diagonal-shifted frame centers stay in bounds
#     generator_guarantees:
#       centers are chosen from the near-middle cells of the original size
#       formula, which guarantees at least three diagonal layers and therefore
#       multiple equal-sized frames in every accepted sample
#   placement_legality:
#       each frame is drawn only when its full bounding box is in bounds
#       because the generator keeps the cross near the middle, no candidate
#       center causes clipping or ambiguous half-frames
#   observability:
#       across 2-4 examples, a human sees the same cross input type but outputs
#       whose boxes all share one fixed size while being repeated on a diagonal
#       lattice; this is visually distinct from both the original spiral and the
#       single-center expanding-frame D6 variant
#   cleanliness:
#       there is only one foreground color and one object family (square frames),
#       so overlap remains readable rather than noisy


def _find_cross(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]], int]:
    """Return center-hole row/col, directional arm cells, and common arm length."""
    h, w = grid.shape
    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == 8:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    arm_len = max((len(cells) for _, cells in arms), default=0)
    return cr, cc, arms, arm_len


def _frame_fits(h: int, w: int, center_r: int, center_c: int, arm_len: int) -> bool:
    return (
        center_r - arm_len >= 0
        and center_c - arm_len >= 0
        and center_r + arm_len < h
        and center_c + arm_len < w
    )


def _frame_centers(
    h: int, w: int, cr: int, cc: int, arm_len: int
) -> list[tuple[int, int]]:
    """
    Enumerate the fixed-size frame centers.

    Layer 0 uses the cross center itself. Every later layer repeats the same
    frame on the four diagonal offsets at stride 2.
    """
    centers: list[tuple[int, int]] = []
    layer = 0
    while True:
        if layer == 0:
            layer_centers = [(cr, cc)]
        else:
            offset = 2 * layer
            layer_centers = [
                (cr - offset, cc - offset),
                (cr - offset, cc + offset),
                (cr + offset, cc - offset),
                (cr + offset, cc + offset),
            ]

        valid = [
            (r, c)
            for r, c in layer_centers
            if _frame_fits(h, w, r, c, arm_len)
        ]
        if not valid:
            break
        centers.extend(valid)
        layer += 1
    return centers


def _draw_frame(out: np.ndarray, center_r: int, center_c: int, arm_len: int) -> None:
    top = center_r - arm_len
    bottom = center_r + arm_len
    left = center_c - arm_len
    right = center_c + arm_len
    out[top, left : right + 1] = 8
    out[bottom, left : right + 1] = 8
    out[top : bottom + 1, left] = 8
    out[top : bottom + 1, right] = 8


def _layer_count(h: int, w: int, cr: int, cc: int, arm_len: int) -> int:
    count = 0
    layer = 0
    while True:
        if layer == 0:
            layer_centers = [(cr, cc)]
        else:
            offset = 2 * layer
            layer_centers = [
                (cr - offset, cc - offset),
                (cr - offset, cc + offset),
                (cr + offset, cc - offset),
                (cr + offset, cc + offset),
            ]
        if not any(_frame_fits(h, w, r, c, arm_len) for r, c in layer_centers):
            break
        count += 1
        layer += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the teal cross with a repeated lattice of fixed-size rectangular frames.

    Input structure:
    - bg = 0 (black). A single teal (8) cross has four equal cardinal arms and an empty center cell.

    Rule:
    - Locate the cross center from the densest teal row and column, then read the common arm length.
    - The arm tips define one square frame of side 2*arm_len+1 around the center.
    - Draw that same frame at the cross center and again at diagonal offsets of 2, 4, 6, ... cells in all four sign combinations, but only when the whole frame stays in bounds.

    Key insight: D6 turns the spiral into separate closed boxes, and D18 freezes the segment growth, so the output is no longer expanding rings but a repeated equal-size box template copied on a 2-spaced diagonal lattice.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    cr, cc, _arms, arm_len = _find_cross(grid)
    if arm_len == 0:
        return out

    for center_r, center_c in _frame_centers(h, w, cr, cc, arm_len):
        _draw_frame(out, center_r, center_c, arm_len)

    return out


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
MAX_OUTER_RETRIES = 200
CROSS_COLOR = 8
MIN_LAYER_COUNT = 3
MIN_FRAME_COUNT = 9
FILL_LO, FILL_HI = 0.42, 0.58
# --- end CFG -----------------------------------------------------------


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    """Build a square grid with a plus-sign cross (center hole) at (cr, cc)."""
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _candidate_centers(size: int, arm: int) -> list[tuple[int, int]]:
    """
    Near-central centers that preserve several diagonal frame layers.

    The combo only reads clearly when multiple same-sized boxes fit, so we keep
    the cross near the middle and reject placements with too few diagonal copies.
    """
    mids = [size // 2 - 1, size // 2]
    candidates = []
    for cr in mids:
        for cc in mids:
            layer_count = _layer_count(size, size, cr, cc, arm)
            frame_count = len(_frame_centers(size, size, cr, cc, arm))
            if layer_count >= MIN_LAYER_COUNT and frame_count >= MIN_FRAME_COUNT:
                candidates.append((cr, cc))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    """Pick an arm length and a legal near-central placement, then validate the sample."""
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    candidates = _candidate_centers(size, arm)
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        layer_count = _layer_count(size, size, cr, cc, arm)
        frame_count = len(_frame_centers(size, size, cr, cc, arm))
        fill = float(np.mean(out == CROSS_COLOR))
        if layer_count >= MIN_LAYER_COUNT and frame_count >= MIN_FRAME_COUNT:
            if FILL_LO <= fill <= FILL_HI and np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
