# Combo variant of aa4ec2a5 (index 38)
# combo_spec: D2/2+D14/0
# Name: 凸形状对象+反转着色条件

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: detect whether a blue 4-connected object encloses a background hole.
#      Hole-bearing objects turn teal, the hole itself turns magenta, and the red outline is
#      drawn around the hole-filled solid mask.
#    - Combo task: every object is convex, so holes cannot exist at all. The classification
#      cue therefore moves from "has a hole" to "how much of its tight bbox is filled".
#      We use the visible ratio area / bbox_area with threshold 3/4.
#      Example:
#        * a 5x6 rectangle has ratio 30/30 and satisfies the ratio condition, so it stays
#          blue and only gets the red outline.
#        * a 7x7 diamond has ratio 25/49 and fails the ratio condition, so the whole object
#          turns teal before receiving the same red outline.
#      Relative to the original, the changed branch is not the outline logic but the per-
#      component test that decides which convex shapes are recolored.
# b) Complexity calibration
#    - To keep the rule inferable, generation uses only a small set of clearly convex shape
#      families with repeated support:
#        dense class  : rectangles and shallow chamfered hexagons, all comfortably above 3/4
#        sparse class : diamonds and centered triangles, all comfortably below 3/4
#    - Every grid contains both classes, so a solver cannot explain the teal marks by size,
#      by object count, or by an accidental one-off geometry.
# c) Implicit-assumption audit for main()
#    - The original `_interior_holes()` branch assumed some objects may have enclosed holes.
#      That assumption is intentionally removed; `main()` must no longer depend on any hole
#      scan or on magenta writes.
#    - `_meets_fill_ratio()` assumes each component is processed in its own tight bbox.
#      `find_components(..., monochromatic=True)` still guarantees that.
#    - `_outline_mask()` originally used a hole-filled solid mask. In this combo the object
#      mask itself is already the full solid footprint because convex generated objects have
#      no holes, so outlining the component mask is correct.
# d) Placement legality
#    - The generator preserves the original border clearance and Chebyshev >= 3 spacing, so
#      components do not touch, outlines never collide, and all writes stay in-bounds.
# e) Visual observability
#    - Humans can see the ratio cue directly: boxier convex objects stay blue, while visibly
#      thinner convex objects inside similarly sized bboxes turn teal. Because each sample
#      contains both behaviors, 2-4 examples are enough to infer the thresholded logic.
# f) Visual cleanliness
#    - The output uses only four colors: yellow background, blue untouched objects, teal
#      recolored objects, and red outlines. No hole color is used, so there is less layering
#      than in the source task and each object remains easy to parse.
# g) Color-change inferability
#    - Teal is tied to a structural measurement, not to arbitrary identity. The generator
#      keeps ratios away from the 3/4 boundary, so the examples consistently show a clean
#      separation between high-fill convex objects and low-fill convex objects.
# h) Post-change input assumptions
#    - `main()` now assumes components are hole-free convex masks whose bbox fill ratio is a
#      meaningful classifier. `generate_input()` enforces 4-connectivity, no holes, row/col
#      convexity, and the presence of at least one above-threshold and one below-threshold
#      component, so these new assumptions are constructed rather than left accidental.

import random
import sys
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    candidate = parent / "solvers" / "arc_utils.py"
    if candidate.exists():
        sys.path.insert(0, str(candidate.parent))
        break

from arc_utils import components_mask, find_components, get_bg, is_contiguous


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


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


FILL_RATIO_NUM = 3
FILL_RATIO_DEN = 4


def _bbox_area(mask: np.ndarray) -> int:
    return int(mask.shape[0] * mask.shape[1])


def _bbox_area_from_coords(coords: np.ndarray) -> int:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _fill_ratio(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(_bbox_area(mask))


def _meets_fill_ratio(mask: np.ndarray) -> bool:
    return int(mask.sum()) * FILL_RATIO_DEN >= _bbox_area(mask) * FILL_RATIO_NUM


def _meets_fill_ratio_coords(coords: np.ndarray) -> bool:
    return int(coords.shape[0]) * FILL_RATIO_DEN >= _bbox_area_from_coords(coords) * FILL_RATIO_NUM


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around convex objects, and mark the lower-fill ones by bbox ratio.

    Input structure:
    - A mostly uniform background contains 3-4 blue 4-connected convex objects.
    - Objects have no enclosed holes; each one is a solid convex footprint inside its own bbox.

    Rule:
    - For every foreground component, draw a one-cell 8-neighborhood outline in red (2).
    - Compute each component's fill ratio as object area divided by tight bbox area.
    - If the ratio is below 3/4, recolor that component to teal (8); otherwise keep it blue (1).

    Key insight: The original hole test disappears because convex shapes cannot contain closed
    cavities here; the deciding cue is instead how densely the object fills its bounding box.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)

        if not _meets_fill_ratio_coords(coords):
            out[mask] = 8

        outline = _outline_mask(mask)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (c) ---------------

GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
LOW_FILL_COUNT_RANGE = (1, 2)

FAMILY_RETRIES = 40
GEN_OUTER_RETRIES = 300
PLACEMENT_RETRIES = 300

CHEBYSHEV_MIN = 3
BORDER_MIN = 1

RECT_H_RANGE = (2, 7)
RECT_W_RANGE = (2, 8)

HEX_H_RANGE = (4, 8)
HEX_W_RANGE = (5, 10)
HEX_TRIM_MAX = 2

DIAMOND_HW_CHOICES = [5, 7, 9]

TRI_H_RANGE = (4, 8)
TRI_W_RANGE = (5, 9)

PASS_RATIO_MIN = 0.82
FAIL_RATIO_MAX = 0.68
MIN_RECOLORED_AREA = 8
MIN_UNCHANGED_AREA = 8


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


def _is_interval_convex(mask: np.ndarray) -> bool:
    for row in mask:
        cols = np.flatnonzero(row)
        if cols.size and cols[-1] - cols[0] + 1 != cols.size:
            return False
    for col in mask.T:
        rows = np.flatnonzero(col)
        if rows.size and rows[-1] - rows[0] + 1 != rows.size:
            return False
    return True


def _is_valid_convex_component(mask: np.ndarray) -> bool:
    cropped = _crop_mask(mask)
    if cropped is None:
        return False
    return (
        bool(cropped.any())
        and is_contiguous(cropped.astype(int), background=0, connectivity=4)
        and not _has_interior_hole(cropped)
        and _is_interval_convex(cropped)
    )


def _rotate_mask(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(mask, k=int(quarter_turns) % 4).copy()


def _make_rectangle() -> np.ndarray:
    h = random.randint(*RECT_H_RANGE)
    w = random.randint(*RECT_W_RANGE)
    return np.ones((h, w), dtype=bool)


def _make_hexagon(horizontal: bool) -> np.ndarray:
    h = random.randint(*HEX_H_RANGE)
    w = random.randint(*HEX_W_RANGE)
    trim_limit = min(HEX_TRIM_MAX, (h - 1) // 2, (w - 2) // 2)
    if trim_limit < 1:
        return _make_rectangle()

    trim = random.randint(1, trim_limit)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        edge_dist = min(r, h - 1 - r)
        cut = max(0, trim - edge_dist)
        mask[r, cut : w - cut] = True

    if horizontal:
        return mask
    return mask.T.copy()


def _make_diamond() -> np.ndarray:
    h = random.choice(DIAMOND_HW_CHOICES)
    w = random.choice(DIAMOND_HW_CHOICES)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ry = h / 2.0
    rx = w / 2.0

    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            score = abs(r - cy) / ry + abs(c - cx) / rx
            if score <= 1.0 + 1e-9:
                mask[r, c] = True
    return mask


def _make_triangle() -> np.ndarray:
    h = random.randint(*TRI_H_RANGE)
    w = random.randint(*TRI_W_RANGE)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        width = 1 + int(round((w - 1) * r / max(1, h - 1)))
        left = (w - width) // 2
        mask[r, left : left + width] = True
    return _rotate_mask(mask, random.randint(0, 3))


def _make_dense_component() -> np.ndarray | None:
    builders = [
        _make_rectangle,
        lambda: _make_hexagon(True),
        lambda: _make_hexagon(False),
    ]
    for _ in range(FAMILY_RETRIES):
        mask = random.choice(builders)()
        ratio = _fill_ratio(mask)
        if (
            ratio >= PASS_RATIO_MIN
            and _meets_fill_ratio(mask)
            and _is_valid_convex_component(mask)
        ):
            return mask
    return None


def _make_sparse_component() -> np.ndarray | None:
    builders = [_make_diamond, _make_triangle]
    for _ in range(FAMILY_RETRIES):
        mask = random.choice(builders)()
        ratio = _fill_ratio(mask)
        if (
            ratio <= FAIL_RATIO_MAX
            and not _meets_fill_ratio(mask)
            and _is_valid_convex_component(mask)
        ):
            return mask
    return None


def _chebyshev_conflict(
    rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray
) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_low_fill = random.randint(
        LOW_FILL_COUNT_RANGE[0], min(LOW_FILL_COUNT_RANGE[1], num_components - 1)
    )

    indices = list(range(num_components))
    random.shuffle(indices)
    low_fill_indices = set(indices[:num_low_fill])

    components: list[np.ndarray] = []
    low_fill_areas: list[int] = []
    high_fill_areas: list[int] = []

    for i in range(num_components):
        mask = _make_sparse_component() if i in low_fill_indices else _make_dense_component()
        if mask is None:
            return None
        components.append(mask)
        if i in low_fill_indices:
            low_fill_areas.append(int(mask.sum()))
        else:
            high_fill_areas.append(int(mask.sum()))

    if max(low_fill_areas, default=0) < MIN_RECOLORED_AREA:
        return None
    if max(high_fill_areas, default=0) < MIN_UNCHANGED_AREA:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

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

    comps = find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True)
    if len(comps) != num_components:
        return None

    observed_low_fill = 0
    observed_high_fill = 0
    for comp in comps:
        mask = _coords_mask(grid.shape, comp["coords"])
        if not _is_valid_convex_component(mask):
            return None
        if _meets_fill_ratio_coords(comp["coords"]):
            observed_high_fill += 1
        else:
            observed_low_fill += 1

    if observed_low_fill < 1 or observed_high_fill < 1:
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
