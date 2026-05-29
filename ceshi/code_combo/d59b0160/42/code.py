# Combo variant of d59b0160 (index 42)
# combo_spec: D1/0+D2/4+D17/2
# Name: ANY匹配+匹配分量压缩+最小分量种子
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior / trace:
#    - The top-left 4x4 legend still provides 3 target colors.
#    - External objects are still isolated 4-connected rectangles of non-bg cells.
#    - Matching changes from ALL to ANY: a component is selected as soon as it shares
#      at least one legend color.
#    - Among the selected components, the uniquely smallest one is kept intact as the
#      seed; every other selected component is cleared and replaced by one centroid
#      marker pixel in its dominant color. Unselected components stay unchanged.
#    - Example: if the legend shows {1,4,8}, then components with colors {0,1},
#      {0,4,9}, {0,1,4,8}, {0,5}. Under the base task only {0,1,4,8} is removed.
#      Here the first three all match by ANY; the smallest of those three remains as a
#      full rectangle, while the other two collapse to single pixels.
# b) Complexity calibration:
#    - The generator forces at least 3 matching components and at least 2 of them are
#      partial-hit (ANY-but-not-ALL), so the rule cannot visually collapse back to the
#      base task.
#    - The seed is sampled from a small area range and other matches are strictly larger,
#      making the "one intact small component + several points" pattern readable.
# c) Hidden-assumption audit:
#    - Under ANY, the legend itself would falsely count as a match if we scanned all
#      non-bg components, so solver and generator both exclude the legend square from
#      component analysis.
#    - The old "treat all matches the same" loop no longer works: we must first collect
#      all matches, choose the smallest seed, then compress only the rest.
#    - Compression relies on a stable marker color. Generated matched rectangles keep
#      color 0 as a unique dominant color by capping sprinkled density below 50%.
# d) Placement legality:
#    - Every patch is placed with 1-cell padding via random_free_location, so components
#      do not overlap or touch each other or the legend.
#    - Compression markers land inside the source component footprint, so output points
#      cannot collide in valid generated samples.
# e) Visual observability:
#    - Each sample contains several selected objects but only one intact selected object.
#      This makes the fused behavior distinguishable from plain deletion and from pure
#      compression.
#    - Partial-hit selected components guarantee that ANY, not ALL, is required.
# f) Cleanliness:
#    - The output contains only three visual categories: unchanged objects, compressed
#      singleton markers, and one intact small seed. No extra overlays or recolor bands.
# g) Inferability:
#    - No ordering depends on hidden color ids; the only ranking is visible area among
#      matched components. The generator enforces a unique smallest matched area so
#      examples do not depend on unspoken tie-break rules.

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
    Detect the smallest top-left square whose bottom edge and right edge share one
    non-background border color. Return (size, border_color, interior target colors).
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        border_color = int(g[sz - 1, sz - 1])
        if border_color == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == border_color) and np.all(g[0:sz, sz - 1] == border_color):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(border_color)
            return int(sz), int(border_color), colors
    raise AssertionError("no legend border found in grid")


def _marker_position(pts: np.ndarray) -> tuple[int, int]:
    """Rounded centroid, snapped to the nearest component cell when necessary."""
    pts = np.asarray(pts, dtype=int)
    mean_r = float(pts[:, 0].mean())
    mean_c = float(pts[:, 1].mean())
    rr = int(math.floor(mean_r + 0.5))
    cc = int(math.floor(mean_c + 0.5))

    exact = np.where((pts[:, 0] == rr) & (pts[:, 1] == cc))[0]
    if exact.size:
        r, c = pts[int(exact[0])]
        return int(r), int(c)

    diffs = pts.astype(float) - np.array([mean_r, mean_c], dtype=float)
    dist2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    best = np.where(dist2 == dist2.min())[0]
    tied = pts[best]
    order = np.lexsort((tied[:, 1], tied[:, 0]))
    r, c = tied[int(order[0])]
    return int(r), int(c)


def _dominant_color(g: np.ndarray, pts: np.ndarray) -> int:
    """Most frequent component color; ties fall back to the color nearest the centroid."""
    pts = np.asarray(pts, dtype=int)
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


def _external_components(g: np.ndarray, bg: int, legend_size: int) -> list[np.ndarray]:
    """Return 4-connected non-background components outside the legend square."""
    mask = (g != bg).copy()
    mask[:legend_size, :legend_size] = False
    return components_mask(mask, connectivity=4)


def _component_infos(grid: np.ndarray) -> tuple[int, int, set[int], list[dict]]:
    """Collect reusable component metadata for solver checks and generator validation."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, border_color, legend_colors = _find_legend(g, bg)
    infos = []
    for pts in _external_components(g, bg, legend_size):
        cols = {int(g[r, c]) for r, c in pts}
        infos.append(
            {
                "pts": pts,
                "area": int(len(pts)),
                "top": int(pts[:, 0].min()),
                "left": int(pts[:, 1].min()),
                "colors": cols,
                "any_match": bool(cols & legend_colors),
                "all_match": legend_colors.issubset(cols),
            }
        )
    return legend_size, border_color, legend_colors, infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the smallest ANY-matching component intact and compress the rest to points.

    Input structure:
    - One background color fills most cells.
    - A small legend square sits in the top-left corner; its bottom edge and right edge
      share one border color, and its interior lists the target colors.
    - Elsewhere, separated 4-connected multicolor components occupy the background.

    Rule:
    - Read the target color set from the legend interior.
    - Ignore the legend itself and collect every external component whose color set
      contains at least one legend color.
    - Among those ANY-matching components, keep the smallest one unchanged as the seed.
    - Compress every other matching component to a single centroid pixel colored by that
      component's dominant color; components with no legend color stay unchanged.

    Key insight:
    - The legend is now an overlap trigger instead of a full-palette test, and the
      matched set is split by size: one smallest survivor remains whole while the rest
      collapse into isolated summary dots.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    matches: list[tuple[int, int, int, np.ndarray]] = []
    for pts in _external_components(g, bg, legend_size):
        cols = {int(g[r, c]) for r, c in pts}
        if cols & legend_colors:
            matches.append((int(len(pts)), int(pts[:, 0].min()), int(pts[:, 1].min()), pts))

    if not matches:
        return out

    seed_idx = min(range(len(matches)), key=lambda i: matches[i][:3])
    for idx, (_area, _top, _left, pts) in enumerate(matches):
        if idx == seed_idx:
            continue
        marker_r, marker_c = _marker_position(pts)
        marker_color = _dominant_color(g, pts)
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

DENSITY_LO = 0.10
DENSITY_HI = 0.34
MATCHED_DENSITY_HI = 0.24

MATCH_COUNTS = [3, 4]
MATCH_WEIGHTS = [4, 1]
NUM_KEPT = 2
MAX_RETRIES = 500

SEED_DIMS = [(3, 3), (3, 4), (4, 3)]
KEEP_H = (3, 6)
KEEP_W = (3, 6)
TALL_H = (5, 8)
TALL_W = (2, 3)
WIDE_H = (3, 4)
WIDE_W = (5, 7)
MED_H = (4, 6)
MED_W = (4, 6)
MATCH_SHAPE_WEIGHTS = [5, 2, 2]  # medium, tall, wide


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the top-left legend in both input and output."""
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


def _sample_mode_dims(weights: list[int]) -> tuple[int, int]:
    """Sample rectangle dimensions from medium / tall / wide modes."""
    mode = random.choices(["medium", "tall", "wide"], weights=weights)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _seed_dims() -> tuple[int, int]:
    """Pick a small seed rectangle."""
    return random.choice(SEED_DIMS)


def _larger_match_dims(seed_area: int) -> tuple[int, int]:
    """Pick a matched rectangle strictly larger than the seed."""
    for _ in range(100):
        h, w = _sample_mode_dims(MATCH_SHAPE_WEIGHTS)
        if h * w > seed_area:
            return h, w
    return 4, 4


def _keep_dims() -> tuple[int, int]:
    """Pick a moderate unmatched rectangle."""
    return random.randint(*KEEP_H), random.randint(*KEEP_W)


def _sample_colored_count(area: int, min_required: int, hi_ratio: float) -> int:
    lo = max(1, math.ceil(area * DENSITY_LO), min_required)
    hi = max(lo, min(area - 1, math.floor(area * hi_ratio)))
    return random.randint(lo, hi)


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """
    Build a solid rectangular component with sparse colored pixels over dominant color 0.

    Roles:
    - keep: contains no target colors
    - seed_partial: smallest ANY-match, contains 1 or 2 target colors
    - compress_partial: matched and compressed, contains 1 or 2 target colors
    - compress_full: matched and compressed, contains all target colors
    """
    area = h * w
    non_targets = [v for v in PALETTE if v not in targets]

    if role == "keep":
        forced = []
        allowed = non_targets
        n_colored = _sample_colored_count(area, 1, DENSITY_HI)
    elif role in {"seed_partial", "compress_partial"}:
        hit_count = random.randint(1, NUM_TARGETS - 1)
        chosen_targets = random.sample(targets, hit_count)
        forced = list(chosen_targets)
        allowed = non_targets + chosen_targets
        n_colored = _sample_colored_count(area, hit_count, MATCHED_DENSITY_HI)
    elif role == "compress_full":
        forced = list(targets)
        allowed = PALETTE
        n_colored = _sample_colored_count(area, NUM_TARGETS, MATCHED_DENSITY_HI)
    else:
        raise ValueError(f"unknown role: {role}")

    patch = np.zeros((h, w), dtype=int)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    colors = forced + [random.choice(allowed) for _ in range(n_colored - len(forced))]
    random.shuffle(colors)
    for (r, c), color in zip(coords, colors):
        patch[r, c] = color
    return patch


def _patch_marker(patch: np.ndarray) -> tuple[int, int, int]:
    """Return the local marker row, col, color after compression."""
    pts = np.argwhere(patch != BG)
    r, c = _marker_position(pts)
    color = _dominant_color(patch, pts)
    return int(r), int(c), int(color)


def _component_specs(targets: list[int]) -> list[dict]:
    """Create one seed, several matched-to-compress components, and two unmatched ones."""
    match_count = random.choices(MATCH_COUNTS, weights=MATCH_WEIGHTS)[0]
    seed_h, seed_w = _seed_dims()
    seed_area = seed_h * seed_w

    specs = [
        {
            "role": "seed_partial",
            "patch": _component_patch(seed_h, seed_w, targets, "seed_partial"),
            "area": seed_area,
        },
        {
            "role": "compress_partial",
            "patch": _component_patch(*_larger_match_dims(seed_area), targets, "compress_partial"),
            "area": None,
        },
    ]
    specs[1]["area"] = int(specs[1]["patch"].shape[0] * specs[1]["patch"].shape[1])

    for _ in range(match_count - 2):
        role = random.choices(["compress_partial", "compress_full"], weights=[3, 1])[0]
        h, w = _larger_match_dims(seed_area)
        specs.append({"role": role, "patch": _component_patch(h, w, targets, role), "area": h * w})

    for _ in range(NUM_KEPT):
        h, w = _keep_dims()
        specs.append({"role": "keep", "patch": _component_patch(h, w, targets, "keep"), "area": h * w})
    return specs


def _place_components(inp: np.ndarray, out: np.ndarray, specs: list[dict]) -> bool:
    """Place all components with 1-cell separation and build the expected output."""
    for spec in sorted(specs, key=lambda x: x["area"], reverse=True):
        patch = spec["patch"]
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False

        paste(inp, patch, r, c)
        if spec["role"] in {"keep", "seed_partial"}:
            paste(out, patch, r, c)
        else:
            mr, mc, color = _patch_marker(patch)
            out[r + mr, c + mc] = color
    return True


def _behavior_signature(grid: np.ndarray) -> dict:
    """Summarize the fused-rule-relevant properties of one generated input."""
    _legend_size, _border_color, legend_colors, infos = _component_infos(grid)
    any_infos = [info for info in infos if info["any_match"]]
    all_infos = [info for info in infos if info["all_match"]]
    partial_infos = [info for info in infos if info["any_match"] and not info["all_match"]]
    areas = sorted(info["area"] for info in any_infos)
    return {
        "legend_colors": legend_colors,
        "external_count": len(infos),
        "any_match_count": len(any_infos),
        "all_match_count": len(all_infos),
        "partial_any_count": len(partial_infos),
        "min_any_area": areas[0] if areas else None,
        "second_any_area": areas[1] if len(areas) > 1 else None,
    }


def _gen_attempt() -> np.ndarray | None:
    """Build one valid fused-variant input, or return None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)
    specs = _component_specs(targets)

    if not _place_components(inp, out, specs):
        return None

    sig = _behavior_signature(inp)
    if sig["any_match_count"] < 3:
        return None
    if sig["partial_any_count"] < 2:
        return None
    if sig["second_any_area"] is None or sig["min_any_area"] >= sig["second_any_area"]:
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
