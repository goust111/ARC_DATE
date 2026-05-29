# Combo variant of da515329 (index 11)
# combo_spec: D3/1+D6/1
# Name: 同心矩形框 + 每圈递增颜色

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the combo variant
# -------------------------------------
# spec_combo:
#   preserved_input:
#     grid: square black canvas
#     object: exactly one teal hollow-centered cross with four equal cardinal arms
#     arm_len: 1..4, inherited from the base task
#   fused_output:
#     geometry:
#       replace the continuous spiral corridor with disconnected concentric
#       rectangular frames centered on the cross hole
#       the innermost frame is the tip-bbox [cr-arm, cr+arm] x [cc-arm, cc+arm]
#       each later frame expands outward by 2 cells on all sides, leaving a
#       1-cell black gap between neighboring frames
#     coloring:
#       frame 1 uses color 1, frame 2 uses color 2, frame 3 uses color 3, ...
#       color depends only on visible frame depth, not on draw order artifacts
#   concrete_behavior_trace:
#     example_arm_2:
#       input tips lie on a 5x5 bbox around the empty center
#       base task: four agents leave the tips, turn clockwise, and eventually
#       weave one connected spiral-like maze in teal
#       combo task: first complete that 5x5 perimeter in color 1, then draw the
#       9x9 perimeter in color 2, then 13x13 in color 3, etc., with black gaps
#       kept intact; no bridging segments are drawn between rectangles
#   complexity_calibration:
#     shape variation still comes from arm length and center placement, so samples
#     are not identical, but the rule family stays low-entropy and human-readable
#     generator requires several visible rings so both "discrete frames" and
#     "outer rings use larger color ids" are repeatedly supported in each sample
#   implicit_assumption_audit:
#     removed_assumptions:
#       all spiral-specific assumptions disappear: turtle headings, turn priority,
#       step lengths, occupancy spacing checks, parity cleanup, border nub repair
#     new_solver_assumptions:
#       the teal cells form one axis-aligned center-hole cross
#       the densest teal row/column still identify the center hole
#       all four arms have the same length, so a single arm_len defines the seed
#       frame; generator enforces this by construction
#     fusion_conflict_check:
#       D3 originally recolored a finished occupancy mask by radius buckets
#       D6 changes the occupancy itself into discrete frames
#       the consistent fusion is to color each actual frame directly by its frame
#       index; this preserves "layer -> color" while matching the new geometry
#   placement_legality:
#     every frame rectangle is drawn only if fully in bounds
#     padding grows by 2, so successive perimeters never overlap and keep a
#     one-cell black moat
#     generator only chooses centers with enough margin for at least 3 rings
#   observability:
#     from a few samples, a human can see two coupled facts:
#       outputs are nested boxes rather than a continuous spiral
#       colors increase monotonically from inner frame to outer frame
#     no hidden ordering or non-visual metadata is used
#   visual_cleanliness:
#     each cell belongs to at most one frame, backgrounds remain black, and
#     each color occupies long straight edge runs, so the pattern stays legible
#   color_inference:
#     under the inherited size formula, visible ring counts stay small enough
#     that colors remain within ARC palette ids 1..8


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


def _frame_rectangles(h: int, w: int, cr: int, cc: int, arm_len: int) -> list[tuple[int, int, int, int]]:
    """List all fully in-bounds concentric frame rectangles."""
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


def _draw_frame(out: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    top, left, bottom, right = rect
    out[top, left : right + 1] = color
    out[bottom, left : right + 1] = color
    out[top : bottom + 1, left] = color
    out[top : bottom + 1, right] = color


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
    Convert a teal hollow cross into spaced concentric frames whose colors grow by layer.

    Input structure:
    - Background is black (0). The only object is one teal (8) plus sign with an empty center cell.
    - The four equal arms identify a shared center and four tips that define the seed square.

    Rule:
    - Find the cross center from the densest teal row and column, then read the common arm length.
    - Use the tip bounding box as the first rectangular frame; enlarge that rectangle by 2 cells per side each time while it stays fully inside the grid.
    - Draw only the perimeter of each rectangle, keeping every frame 1 cell thick with a 1-cell black gap, and color frame k with color k starting from 1 at the inside.

    Key insight: The fused rule replaces the spiral trajectory with its discrete rectangular envelopes, and the depth of those envelopes directly determines the output color.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    cr, cc, _arms, arm_len = _find_cross(grid)
    if arm_len == 0:
        return grid.copy()

    out = np.zeros((h, w), dtype=int)
    for layer, rect in enumerate(_frame_rectangles(h, w, cr, cc, arm_len), start=1):
        _draw_frame(out, rect, layer)
    return out


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
MAX_OUTER_RETRIES = 200
CROSS_COLOR = 8
MIN_RING_COUNT = 3
MIN_COLOR_COUNT = 3
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


def _ring_count(size: int, arm: int, cr: int, cc: int) -> int:
    min_margin = min(cr, size - 1 - cr, cc, size - 1 - cc)
    if min_margin < arm:
        return 0
    return 1 + (min_margin - arm) // 2


def _candidate_centers(size: int, arm: int) -> list[tuple[int, int]]:
    """
    Return legal centers with enough margin for several visible rings.

    Restricting to the central band avoids heavily truncated, one-sided layouts
    while still leaving enough placement variation across samples.
    """
    min_margin = arm + 2 * (MIN_RING_COUNT - 1)
    mid = (size - 1) / 2.0
    candidates = []
    for cr in range(size):
        for cc in range(size):
            if min(cr, size - 1 - cr, cc, size - 1 - cc) < min_margin:
                continue
            if abs(cr - mid) > 2 or abs(cc - mid) > 2:
                continue
            candidates.append((cr, cc))
    return candidates


def _is_valid_sample(inp: np.ndarray, out: np.ndarray, arm: int, cr: int, cc: int) -> bool:
    ring_count = _ring_count(inp.shape[0], arm, cr, cc)
    colors = sorted(int(x) for x in np.unique(out) if x != 0)
    component_count = _count_components(out != 0)
    if ring_count < MIN_RING_COUNT:
        return False
    if len(colors) < MIN_COLOR_COUNT:
        return False
    if colors != list(range(1, ring_count + 1)):
        return False
    if component_count != ring_count:
        return False
    return np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    candidates = _candidate_centers(size, arm)
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        if _is_valid_sample(inp, out, arm, cr, cc):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
