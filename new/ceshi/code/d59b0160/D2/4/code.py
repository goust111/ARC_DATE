# Variant of d59b0160
# Dimension: D2 — 核心变换操作：删除 vs 其他处理 (transform)
# Change: 匹配的连通分量被删除（所有像素替换为背景色7）... → 匹配分量缩小为单像素标记
#
# Design analysis
# ===============
# spec_variant:
#   preserved_structure:
#     grid: 16x16, background color 7
#     legend: fixed 4x4 block in the top-left corner
#     legend_border: bottom row and right column are color 3
#     legend_payload: exactly 3 target colors placed once in the 3x3 interior
#     objects: 4-connected non-background components elsewhere, still drawn as solid
#       rectangles of color 0 with extra colored pixels sprinkled inside
#   core_change_D2:
#     original_behavior: a component containing all legend colors is erased to background
#     new_behavior: such a component is erased first, then replaced by exactly one pixel
#       at its centroid-like position; the marker color is the component's dominant color
#   concrete_trace:
#     - Suppose the legend shows {1,4,8}.
#     - A large rectangle containing colors {0,1,4,8,9} used to disappear completely.
#     - In this variant, the same rectangle collapses to one pixel near its geometric center.
#     - Because the generator keeps matched rectangles fairly large, the input->output change
#       is visually read as compression rather than deletion.
#   complexity_calibration:
#     - The legend rule is unchanged, so the puzzle still hinges on set inclusion.
#     - The transformed objects remain large rectangles to make the compression obvious.
#     - Matched rectangles are generated with a unique dominant color, usually 0, so the
#       marker color is unambiguous and not dependent on invisible tie-breaking.
#   implicit_assumption_audit:
#     - _bg_boundary still works because the outer border remains mostly background 7.
#     - _find_legend still works because the legend border geometry is unchanged.
#     - components_mask(g != bg) still yields whole objects because every object rectangle is
#       filled with non-bg cells.
#     - The old loop wrote bg to every matched pixel. That assumption no longer suffices:
#       after clearing the component we must place exactly one marker pixel back.
#     - A raw rounded centroid can fall outside an irregular component in general, so the
#       solver snaps to the nearest component pixel when needed. The generator keeps objects
#       rectangular, which guarantees the rounded centroid already lies inside for samples.
#   placement_legality:
#     - Objects are still placed with at least one background cell of 8-neighborhood padding,
#       so different components never overlap or touch.
#     - The marker of a compressed component lands inside the component's own footprint, so
#       it cannot collide with another object in valid generated instances.
#   visual_observability:
#     - Every generated grid has at least one matched component.
#     - Matched components are forced to have area >= 12, so shrinking to one pixel is
#       obvious in 2-4 examples.
#     - The output differs from the original task because a non-background residue remains at
#       each matched object's center instead of a blank hole.
#   cleanliness:
#     - The legend remains isolated in the top-left corner.
#     - Components stay separated; each compressed marker sits alone in a cleared rectangle,
#       producing clean, readable outputs.
#   color_inference:
#     - The marker color is derived from internal color frequency, not from color-id mapping.
#     - The generator guarantees a unique mode on matched components, avoiding ambiguous ties.

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the top-left legend whose bottom edge and right edge share one border color.
    Return (legend_size, border_color, target_color_set).
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    assert False, "no legend border found in grid"


def _marker_position(pts: np.ndarray) -> tuple[int, int]:
    """
    Rounded centroid, snapped to the nearest component pixel if rounding lands outside.
    """
    mean_r = float(pts[:, 0].mean())
    mean_c = float(pts[:, 1].mean())
    rr = int(math.floor(mean_r + 0.5))
    cc = int(math.floor(mean_c + 0.5))

    diffs = pts - np.array([rr, cc], dtype=float)
    exact = np.where((diffs[:, 0] == 0) & (diffs[:, 1] == 0))[0]
    if exact.size:
        r, c = pts[int(exact[0])]
        return int(r), int(c)

    diffs = pts.astype(float) - np.array([mean_r, mean_c], dtype=float)
    dist2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    best = np.where(dist2 == dist2.min())[0]
    if best.size == 1:
        r, c = pts[int(best[0])]
        return int(r), int(c)

    tied = pts[best]
    order = np.lexsort((tied[:, 1], tied[:, 0]))
    r, c = tied[int(order[0])]
    return int(r), int(c)


def _marker_color(g: np.ndarray, pts: np.ndarray) -> int:
    """
    Dominant color of the component.
    Ties are broken by geometric centrality; generated samples avoid ties.
    """
    cols = np.array([int(g[r, c]) for r, c in pts], dtype=int)
    vals, cnts = np.unique(cols, return_counts=True)
    best_count = int(cnts.max())
    tied = vals[cnts == best_count]
    if tied.size == 1:
        return int(tied[0])

    mean_r = float(pts[:, 0].mean())
    mean_c = float(pts[:, 1].mean())
    centroid = np.array([mean_r, mean_c], dtype=float)

    best_color = None
    best_metric = None
    best_anchor = None
    for color in tied:
        color_pts = pts[cols == int(color)].astype(float)
        d2 = ((color_pts - centroid) ** 2).sum(axis=1)
        metric = float(d2.mean())
        anchor = tuple(map(int, color_pts[np.argmin(d2)]))
        if (
            best_metric is None
            or metric < best_metric
            or (metric == best_metric and anchor < best_anchor)
        ):
            best_color = int(color)
            best_metric = metric
            best_anchor = anchor
    return int(best_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress matched connected components into single centroid markers.

    Input structure:
    - Most of the board is background.
    - The top-left corner contains a small square legend whose bottom edge and right edge
      share one border color; the interior cells show the target color set.
    - Away from the legend, each object is a 4-connected non-background component.

    Rule:
    - Read the target color set from the legend interior.
    - Inspect every non-background component and test whether it contains all legend colors.
    - Components that fail the test stay unchanged.
    - Components that pass the test are cleared, then replaced by one pixel near their
      centroid; that pixel takes the component's most frequent color.

    Key insight:
    - The legend still identifies which components are selected, but the selected components
      are not deleted outright anymore; each one collapses into a single summary marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if not legend_colors.issubset(cols):
            continue
        marker_r, marker_c = _marker_position(pts)
        marker_color = _marker_color(g, pts)
        for r, c in pts:
            out[r, c] = bg
        out[marker_r, marker_c] = marker_color
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

DENSITY_LO = 0.10
DENSITY_HI = 0.55
MATCHED_DENSITY_HI = 0.33
MATCHED_MIN_AREA = 12

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the top-left legend box and copy it to both input and output."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims(matched: bool) -> tuple[int, int]:
    """Sample component dimensions, enforcing larger matched components for visible compression."""
    for _ in range(50):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if not matched or h * w >= MATCHED_MIN_AREA:
            return h, w
    return 4, 4


def _component_patch(h: int, w: int, targets: list[int], matched: bool) -> np.ndarray:
    """Build an object patch; matched patches are forced to contain all targets and a unique mode."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    patch = np.zeros((h, w), dtype=int)
    non_targets = [v for v in PALETTE if v not in targets]

    if matched:
        lo = max(lo, NUM_TARGETS)
        hi = min(hi, max(lo, math.floor(area * MATCHED_DENSITY_HI)))
        hi = min(hi, area - NUM_TARGETS)
        n_colored = random.randint(lo, hi)
        coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
        colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - NUM_TARGETS)]
        random.shuffle(colors)
        for (r, c), color in zip(coords, colors):
            patch[r, c] = color
        return patch

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    used_target_count = random.randint(0, NUM_TARGETS - 1)
    allowed = non_targets + random.sample(targets, used_target_count)
    colors = [random.choice(allowed) for _ in range(n_colored)]
    for (r, c), color in zip(coords, colors):
        patch[r, c] = color
    return patch


def _patch_marker(patch: np.ndarray) -> tuple[int, int, int]:
    """Return the local-row, local-col, color of the compressed marker for one patch."""
    pts = np.argwhere(patch != BG)
    r, c = _marker_position(pts)
    color = _marker_color(patch, pts)
    return int(r), int(c), int(color)


def _gen_attempt() -> np.ndarray | None:
    """Build one valid input grid or return None if placement/verification fails."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    matched_count = component_count - NUM_KEPT
    roles = [True] * matched_count + [False] * NUM_KEPT
    random.shuffle(roles)

    for matched in roles:
        h, w = _component_dims(matched)
        patch = _component_patch(h, w, targets, matched)
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c)
        if matched:
            mr, mc, color = _patch_marker(patch)
            out[r + mr, c + mc] = color
        else:
            paste(out, patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
