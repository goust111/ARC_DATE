# Variant of da515329
# Dimension: D6 — 路径类型 (path)
# Change: 矩形螺旋路径（沿网格正交方向转90度弯的螺旋）... → 同心矩形框（非连续螺旋）

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   preserved_input:
#     grid: square black canvas with one teal center-hole cross
#     cross: four equal cardinal arms, length 1..4, color 8
#   transformed_output:
#     seed_shape:
#       the four arm tips define the smallest axis-aligned square frame
#       with bbox [cr-arm, cr+arm] x [cc-arm, cc+arm]
#     path_change:
#       original: one connected clockwise spiral corridor
#       variant: disconnected 1-cell-thick rectangular frames
#     frame_growth:
#       expand the seed frame by padding 2 cells at a time
#       draw only the perimeter of each in-bounds rectangle
#       the 2-cell stride leaves a 1-cell gap between adjacent frames
#     visibility_goal:
#       keep at least 3 rings in generated samples so the output is visibly
#       "many separate boxes" rather than "one thick square"
#   concrete_trace:
#     example_arm_2:
#       input tips lie on a 5x5 bbox around the center hole
#       original logic would start four agents from those tips and connect
#       the loops into one self-avoiding spiral
#       variant logic first closes that 5x5 bbox as a frame, then adds 9x9,
#       13x13, ... perimeters while they fit; the bridge segments between
#       loops are never drawn, so each outer ring becomes its own component
#   complexity_calibration:
#     the rule still depends on arm length and available border margin, so
#     outputs vary across examples, but there are no extra colors or object
#     types that would introduce ambiguous correspondences
#   implicit_assumption_audit:
#     old_agent_state:
#       all turn-priority, step-length, spacing, and parity cleanup code
#       depended on the path being a single spiral; none of those assumptions
#       remain valid after discretizing into separate frames
#     new_solver_assumptions:
#       there is exactly one teal center-hole cross
#       the four arms are equal and axis-aligned
#       enough border margin exists to fit several 2-spaced rings
#     generator_guarantees:
#       it places the cross only at near-central cells of the original size
#       formula, which yields >=3 rings and keeps every frame in-bounds
#   placement_legality:
#     frames are derived from the cross bbox, so they never overlap
#     incorrectly with the input object; each new ring is strictly outside
#     the previous one and separated by one black cell
#   observability:
#     humans can distinguish the variant by component structure alone:
#     the output shows nested boxes with black gaps, not one continuous maze
#   cleanliness:
#     only one color is used and rings do not cross, so the grid remains
#     easy to parse even when several layers are present


def _find_cross(grid: np.ndarray) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]], int]:
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


def _frame_rectangles(h: int, w: int, cr: int, cc: int, arm_len: int) -> list[tuple[int, int, int, int]]:
    """List all fully in-bounds concentric frames starting from the tip bbox."""
    rects = []
    pad = 0
    while True:
        top = cr - arm_len - pad
        bottom = cr + arm_len + pad
        left = cc - arm_len - pad
        right = cc + arm_len + pad
        if top < 0 or left < 0 or bottom >= h or right >= w:
            break
        rects.append((top, left, bottom, right))
        pad += 2
    return rects


def _count_components(mask: np.ndarray) -> int:
    """4-connected component count on a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn a teal center-hole cross into spaced concentric rectangular frames.

    Input structure:
    - bg = 0 (black). A single teal (8) cross has four equal arms and an empty center cell.

    Rule:
    - Locate the cross center from the densest teal row/column, then read the four arm tips.
    - The tips define the smallest rectangle; complete that perimeter, then expand the rectangle by 2 cells at a time.
    - Draw only each in-bounds perimeter, keeping every frame 1 cell thick with a 1-cell black gap between successive frames.

    Key insight: The original spiral can be discretized into its nested loop envelopes; once the tip-bbox is known, the whole output is just repeatedly padded rectangular frames.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    cr, cc, _arms, arm_len = _find_cross(grid)
    if arm_len == 0:
        return out

    for top, left, bottom, right in _frame_rectangles(h, w, cr, cc, arm_len):
        out[top, left : right + 1] = 8
        out[bottom, left : right + 1] = 8
        out[top : bottom + 1, left] = 8
        out[top : bottom + 1, right] = 8

    return out


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
MAX_OUTER_RETRIES = 200
CROSS_COLOR = 8
MIN_RING_COUNT = 3
MIN_COMPONENT_COUNT = 3
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
    Near-central centers that preserve at least MIN_RING_COUNT frames.

    Keeping the cross close to the canvas middle makes the concentric-frame
    behavior visually obvious and avoids degenerate one-sided truncation.
    """
    mids = [size // 2 - 1, size // 2]
    out = []
    for cr in mids:
        for cc in mids:
            min_margin = min(cr, size - 1 - cr, cc, size - 1 - cc)
            ring_count = 1 + (min_margin - arm) // 2 if min_margin >= arm else 0
            if ring_count >= MIN_RING_COUNT:
                out.append((cr, cc))
    return out


def _gen_attempt() -> np.ndarray | None:
    """Pick an arm length and a legal near-central placement, then validate the sample."""
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    candidates = _candidate_centers(size, arm)
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        ring_count = len(_frame_rectangles(size, size, cr, cc, arm))
        component_count = _count_components(out == CROSS_COLOR)
        if ring_count >= MIN_RING_COUNT and component_count >= MIN_COMPONENT_COUNT:
            if np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
