# Combo variant of d59b0160 (index 15)
# combo_spec: D11/0+D15/0
# Name: 图例大小可变+图例颜色数量可变
#
# Design analysis
# ===============
# a) Concrete behavior and execution trace:
#    - Original task:
#      the legend is effectively hard-coded as a 3x3 interior in the top-left 4x4 block, and it always
#      contributes exactly 3 target colors. A component is deleted iff it contains all 3.
#    - Fused variant:
#      the legend interior can now be 2x2, 3x3, 4x4, or 5x5, so the full top-left legend block is 3x3
#      through 6x6 including the border. The number of distinct legend colors is no longer fixed; it
#      varies from 2 to 5 and grows with the chosen legend size.
#    - Trace on one concrete sample:
#      legend interior size = 5, so the visible legend block is 6x6.
#      legend colors = {1, 4, 6, 8}.
#      component A colors = {0, 1, 4, 6, 8, 9} -> delete A.
#      component B colors = {0, 1, 4, 8, 9} -> keep B because color 6 is missing.
#      Under the original fixed-3-color assumption, the same grid would be misread as a smaller legend or
#      as a 3-color criterion; the fused solver must first infer the legend extent, then infer how many
#      distinct colors it encodes, and only then test components.
# b) Complexity calibration:
#    - The size/count freedom is not left unconstrained. Generated pairs come from a small set of
#      size/count combinations: (2,2), (3,2), (3,3), (4,3), (4,4), (5,4), (5,5).
#      This keeps the relation human-discernible: larger legends visibly tend to demand more colors.
#    - Every sample still contains both matching and non-matching objects, so the variable legend is an
#      active control signal rather than decorative noise.
#    - The legend border remains a simple bottom-row + right-column L shape in the top-left corner, so the
#      extra variability is concentrated in two observable axes only: border size and number of swatches.
# c) Implicit-assumption audit of the original solver:
#    - `_bg_boundary(...)` still holds because the outer frame remains dominated by background 7.
#    - The old `_find_legend(...)` assumed "the first valid size is the true legend". That is no longer a
#      safe assumption once larger legends exist, because a partial smaller square could accidentally look
#      valid. The fused solver therefore collects all legal top-left candidates and chooses the largest.
#    - The old code treated every non-bg component uniformly. With larger legends and more swatches, the
#      legend area itself is more likely to contain connected non-bg fragments carrying many colors. The
#      solver now explicitly excludes components wholly inside the legend square; the clue region is not a
#      deletable object.
#    - The old generator only reserved the legend's painted cells. That is unsafe for larger legends
#      because objects could spill into blank cells inside the legend square. The new generator reserves
#      the entire legend bbox on a separate placement canvas.
# d) Placement legality:
#    - The legend always occupies the top-left `outer_size x outer_size` square, where `outer_size` is
#      `inner_size + 1`.
#    - Object placement uses `random_free_location(..., padding=1, padding_connectivity=8)` on a dedicated
#      occupancy canvas that already marks the whole legend square as occupied. This guarantees no overlap,
#      no touching, and no intrusion into the legend.
#    - Components are pasted only after a valid bbox location is reserved, so they remain in-bounds.
# e) Visual observability:
#    - Across examples, the left-top legend is visibly larger or smaller, and the number of distinct clue
#      colors inside it also changes. A human can see that the deletion condition becomes stricter when
#      more legend colors are present.
#    - The rule depends only on visible geometry and visible color membership. It does not rely on numeric
#      color ordering or any hidden metadata.
# f) Visual cleanliness:
#    - The layout still uses sparse separated rectangles on a clean background, with the legend isolated in
#      one corner. There are no overlapping annotation layers.
#    - The size/count variation happens inside a single well-bounded region, so the board does not become
#      cluttered even when the legend grows to 6x6.
# g) Inferability and anti-degeneration guard:
#    - The generator cycles through several size/count regimes instead of silently falling back to the
#      original 3x3 / 3-color case. A quantitative check can therefore verify that non-original regimes
#      dominate a batch of samples.
#    - Because the deletion threshold is literally the set read from the variable-size legend, the two
#      fused controls act on the same inference chain rather than as unrelated cosmetic edits.

from pathlib import Path
import sys

import math
import random

import numpy as np

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    solvers_dir = parent / "solvers"
    if solvers_dir.is_dir():
        sys.path.insert(0, str(solvers_dir))
        break
else:
    raise ImportError("could not locate solvers/ for arc_utils import")

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the top-left legend square whose bottom row and right column share one border color.

    Multiple candidate sizes may exist when the legend is larger, so we keep all legal candidates and
    select the largest one. Returned tuple = (outer_size, border_color, legend_colors).
    """
    h, w = g.shape
    candidates: list[tuple[int, int, set[int]]] = []
    for outer in range(3, min(h, w) + 1):
        border_color = int(g[outer - 1, outer - 1])
        if border_color == bg:
            continue
        if not np.all(g[outer - 1, 0:outer] == border_color):
            continue
        if not np.all(g[0:outer, outer - 1] == border_color):
            continue
        inside = g[0 : outer - 1, 0 : outer - 1]
        colors = set(map(int, np.unique(inside)))
        colors.discard(bg)
        colors.discard(border_color)
        if 2 <= len(colors) <= 5:
            candidates.append((int(outer), border_color, colors))
    assert candidates, "no legend border found in grid"
    return max(candidates, key=lambda item: (item[0], len(item[2])))


def _is_legend_component(pts: np.ndarray, outer: int) -> bool:
    """Return True iff every cell of the component lies inside the legend square."""
    return bool(np.all(pts[:, 0] < outer) and np.all(pts[:, 1] < outer))


def _component_colors(g: np.ndarray, pts: np.ndarray) -> set[int]:
    return {int(g[r, c]) for r, c in pts}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove every non-legend object whose palette covers the full variable legend palette.

    Input structure:
    - One background color fills most of the grid.
    - The top-left corner contains a square legend whose bottom edge and right edge share one border
      color. The legend interior is not fixed: it can be 2x2 through 5x5.
    - Inside that legend interior, a variable number of distinct target colors is shown, from 2 to 5.
    - Elsewhere, separated 4-connected multicolor objects appear on the background.

    Rule:
    - Infer the legend boundary size from the top-left border shape.
    - Read the full set of distinct non-background, non-border colors inside the legend interior.
    - For each 4-connected non-background component outside the legend square, test whether it contains
      every legend color.
    - Erase exactly those components that contain the whole legend set; keep all other components and the
      legend itself unchanged.

    Key insight: the action on objects is unchanged in spirit, but the clue is now doubly variable. The
    solver must recover both how large the legend region is and how many colors it actually contributes
    before it can decide which components satisfy the deletion predicate.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    outer, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        if _is_legend_component(pts, outer):
            continue
        if legend_colors.issubset(_component_colors(g, pts)):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (combo D11/0 + D15/0) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
PALETTE = [1, 2, 4, 5, 6, 8, 9]

LEGEND_CONFIGS = [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5)]
_legend_config_order = LEGEND_CONFIGS[:]
random.shuffle(_legend_config_order)
_legend_config_index = 0

SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (7, 14)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 7)
MED_W = (2, 8)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [4, 5, 6]
LIGHT_COMP_WEIGHTS = [4, 3, 1]
HEAVY_COMP_WEIGHTS = [2, 3, 4]
NUM_KEPT = 3

MAX_RETRIES = 800


def _next_legend_config() -> tuple[int, int]:
    """Cycle through size/count regimes so repeated samples visibly exercise the fusion."""
    global _legend_config_order, _legend_config_index
    if _legend_config_index >= len(_legend_config_order):
        _legend_config_order = LEGEND_CONFIGS[:]
        random.shuffle(_legend_config_order)
        _legend_config_index = 0
    config = _legend_config_order[_legend_config_index]
    _legend_config_index += 1
    return config


def _legend_positions(inner: int, target_count: int) -> list[tuple[int, int]]:
    """Choose spread-out target slots inside the legend interior."""
    cells = [(r, c) for r in range(inner) for c in range(inner)]
    positions: list[tuple[int, int]] = []

    anchor_pool = [(r, c) for r in range(min(2, inner)) for c in range(min(2, inner))]
    positions.append(random.choice(anchor_pool))

    edge_pool = [(r, c) for r, c in cells if r == inner - 1 or c == inner - 1]
    if target_count >= 2 and edge_pool:
        edge_choices = [p for p in edge_pool if p not in positions]
        positions.append(random.choice(edge_choices))

    remaining = [p for p in cells if p not in positions]
    positions.extend(random.sample(remaining, target_count - len(positions)))
    random.shuffle(positions)
    return positions


def _place_legend(
    inp: np.ndarray,
    out: np.ndarray,
    placement: np.ndarray,
    inner: int,
    targets: list[int],
) -> None:
    """Draw a variable-size legend and reserve its full square for later placement."""
    outer = inner + 1
    for g in (inp, out):
        g[:outer, :outer] = BG
        g[outer - 1, :outer] = BORDER
        g[:outer, outer - 1] = BORDER
    placement[:outer, :outer] = BORDER

    for color, (r, c) in zip(targets, _legend_positions(inner, len(targets))):
        inp[r, c] = color
        out[r, c] = color


def _sample_component_count(inner: int, target_count: int) -> int:
    """Use fewer objects for the heavier legend regimes to keep placement reliable."""
    if inner >= 4 or target_count >= 4:
        weights = LIGHT_COMP_WEIGHTS
    else:
        weights = HEAVY_COMP_WEIGHTS
    return random.choices(COMP_COUNTS, weights=weights)[0]


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions according to the original tall/wide/medium bias."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    """Build an object patch: a solid 0 rectangle with colored accents."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if deleted:
        lo = max(lo, len(targets))
    lo = min(lo, area)
    hi = min(area, max(lo, hi))

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
        colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - len(targets))]
    else:
        used_target_count = random.randint(0, len(targets) - 1)
        partial_targets = random.sample(targets, used_target_count)
        allowed = non_targets + partial_targets
        colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _reserve_location(placement: np.ndarray, h: int, w: int) -> tuple[int, int]:
    """Reserve a full rectangle bbox on the placement canvas and return its top-left corner."""
    blocker = np.zeros((h, w), dtype=int)
    r, c = random_free_location(
        placement,
        blocker,
        transparent=BG,
        padding=1,
        padding_connectivity=8,
    )
    paste(placement, blocker, r, c, transparent=BG)
    return r, c


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Small quantitative probe used to verify that the fusion is actually present in samples."""
    bg = _bg_boundary(grid)
    outer, _border, legend_colors = _find_legend(grid, bg)
    nonlegend = [
        pts for pts in components_mask(grid != bg, connectivity=4)
        if not _is_legend_component(pts, outer)
    ]
    matching = sum(legend_colors.issubset(_component_colors(grid, pts)) for pts in nonlegend)
    return {
        "legend_inner_size": outer - 1,
        "target_count": len(legend_colors),
        "matching_components": int(matching),
        "kept_components": int(len(nonlegend) - matching),
    }


def _gen_attempt() -> np.ndarray | None:
    """Build one valid input grid; return None if placement or validation fails."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()
    placement = inp.copy()

    inner, target_count = _next_legend_config()
    targets = random.sample(PALETTE, target_count)
    _place_legend(inp, out, placement, inner, targets)

    component_count = _sample_component_count(inner, target_count)
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted)
        try:
            r, c = _reserve_location(placement, h, w)
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if not deleted:
            paste(out, patch, r, c)

    metrics = _variant_metrics(inp)
    if metrics["matching_components"] < 1 or metrics["kept_components"] < NUM_KEPT:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
