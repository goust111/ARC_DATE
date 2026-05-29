# Combo variant of d59b0160 (index 28)
# combo_spec: D3/0+D7/3+D11/0
# Name: 图例位置不固定+边框颜色不固定+图例大小可变

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import components_mask, paste, random_free_location


"""
Design analysis for the combo variant D3/0 + D7/3 + D11/0

a) Concrete behavior and trace
   - Baseline d59b0160:
     1. Read a fixed 4x4 legend from the top-left corner.
     2. Its border is always green (3), specifically the bottom row and right column.
     3. Its 3x3 interior contains the target palette.
     4. Remove every non-background 4-connected component that contains all legend colors.
   - Fused behavior:
     1. The legend is still a compact square with an L-shaped border, but it is pasted on
        exactly one outer edge, not in a fixed corner.
     2. The interior side length is variable: 2, 3, 4, or 5; total legend window is one
        cell larger in each dimension because the L border occupies one row and one column.
     3. The border color is not fixed. It is the one color that appears only on that L border
        and nowhere in the ordinary objects or in the legend content itself.
     4. The deletion rule is unchanged semantically: ordinary components that realize the full
        legend palette are erased to background.
   - Small trace:
     * Suppose the legend sits on the bottom edge, uses border color 6, has a 4x4 interior,
       and its four swatches are {1,2,4,9}. A component elsewhere containing {0,1,2,4,9}
       is deleted; another containing {0,1,2,9} survives because it misses 4.
     * Original code would fail three ways on the same input: it would only inspect the
       top-left corner, assume border color 3, and assume the legend interior is exactly 3x3.

b) Complexity calibration
   - Edge location uses the four cardinal sides only, but the legend is kept away from the
     four corners. This makes D3 visible in every sample and prevents collapse back to the
     original top-left regime.
   - Interior size is limited to {2,3,4,5}. This is enough to make size inference real while
     keeping the legend compact and readable.
   - The target-palette size equals the interior side length. Larger legends therefore carry
     more required colors, so D11 changes both appearance and deletion strictness.
   - Border color is chosen from the ordinary ARC palette but reserved exclusively for the
     legend border, making D7 visually inferable by elimination rather than by a fixed id.

c) Hidden-assumption audit of the original main()
   - `_find_legend()` assumed `(0,0)` anchoring. Broken: the new solver must scan all four edges.
   - `_find_legend()` assumed one hard-coded orientation (bottom row + right column). Broken:
     once the legend moves to arbitrary edges, the L can be mirrored/rotated so the open side
     faces the touched edge.
   - `_find_legend()` assumed the border color can be read from one fixed corner cell and then
     matched against color 3-like behavior. Broken: border color must now be inferred from a
     unique edge L whose color appears nowhere else.
   - The old code implicitly let the legend participate in the non-background component scan.
     That was fragile even before. The new solver explicitly masks out the whole legend window,
     so legend swatches can never be mistaken for deletable objects.
   - The old size loop started at 2 because total legend size was the searched quantity. Under
     D11, the relevant human-facing quantity is interior size 2..5, so the solver searches
     total sizes 3..6 instead.

d) Placement legality
   - Generator places the legend first, flush to exactly one outer edge and fully in bounds.
   - Ordinary components are placed with `border=1` and `padding=1`, so they stay away from
     the outer frame and from each other. This ensures the edge legend is the only edge object
     and every ordinary object remains one recoverable 4-connected component.
   - Border color is excluded from all ordinary components and from the legend interior.

e) Visual observability
   - A human can see all three control variables directly:
     * edge location from where the compact L sits,
     * border color from the unique unused hue outlining the legend,
     * size from the legend window extent.
   - Because larger legends advertise more colors, samples with different legend sizes produce
     visibly different deletion selectivity rather than a cosmetic-only change.

f) Visual cleanliness
   - Only the legend touches the boundary; everything else stays in the interior. This keeps
     edge scanning clean and prevents clutter between legend signals and ordinary objects.
   - The legend interior contains a small number of isolated swatches on a plain background,
     so the target palette is readable even at the largest 5x5 size.

g) Change inferability
   - No rule depends on invisible numeric ordering of colors. Border color is inferred from its
     structural role and exclusivity, not from its id.
   - The generator uses all four edges, multiple border colors, and all four interior sizes
     across samples, so the fused dimensions are materially active rather than decorative.

h) New solver assumptions and generator guarantees
   - Solver assumptions:
     * there is exactly one edge-attached legend window touching one edge and no corner;
     * its border color appears only on the legend border;
     * ordinary deletable objects lie outside the legend window.
   - Generator guarantees these by construction and validates every sample with `main(inp)`.
"""


GRID_H, GRID_W = 16, 16
BG = 7
BASE_COLOR = 0
EDGE_NAMES = ("top", "bottom", "left", "right")
BORDER_POOL = [1, 2, 3, 4, 5, 6, 8, 9]
MIN_INNER = 2
MAX_INNER = 5

EDGE_TO_ORIENTATIONS = {
    "top": ("br", "bl"),
    "bottom": ("tr", "tl"),
    "left": ("br", "tr"),
    "right": ("bl", "tl"),
}

SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (7, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 7)
MED_W = (2, 8)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 20]
NUM_KEPT = 3
MAX_RETRIES = 2000


def _bg_boundary(g: np.ndarray) -> int:
    """Use the outer frame majority, which remains mostly background."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _legend_layout(
    r0: int, c0: int, inner_size: int, orientation: str
) -> tuple[set[tuple[int, int]], tuple[slice, slice]]:
    sz = int(inner_size) + 1
    border: set[tuple[int, int]] = set()
    if orientation == "br":
        border.update((int(r0 + sz - 1), int(c)) for c in range(c0, c0 + sz))
        border.update((int(r), int(c0 + sz - 1)) for r in range(r0, r0 + sz))
        return border, (slice(r0, r0 + inner_size), slice(c0, c0 + inner_size))
    if orientation == "bl":
        border.update((int(r0 + sz - 1), int(c)) for c in range(c0, c0 + sz))
        border.update((int(r), int(c0)) for r in range(r0, r0 + sz))
        return border, (slice(r0, r0 + inner_size), slice(c0 + 1, c0 + sz))
    if orientation == "tr":
        border.update((int(r0), int(c)) for c in range(c0, c0 + sz))
        border.update((int(r), int(c0 + sz - 1)) for r in range(r0, r0 + sz))
        return border, (slice(r0 + 1, r0 + sz), slice(c0, c0 + inner_size))
    if orientation == "tl":
        border.update((int(r0), int(c)) for c in range(c0, c0 + sz))
        border.update((int(r), int(c0)) for r in range(r0, r0 + sz))
        return border, (slice(r0 + 1, r0 + sz), slice(c0 + 1, c0 + sz))
    raise ValueError(f"unknown legend orientation: {orientation}")


def _iter_edge_windows(h: int, w: int, total_size: int):
    sz = int(total_size)
    for c0 in range(1, w - sz):
        yield "top", 0, int(c0)
        yield "bottom", int(h - sz), int(c0)
    for r0 in range(1, h - sz):
        yield "left", int(r0), 0
        yield "right", int(r0), int(w - sz)


def _find_legend(g: np.ndarray, bg: int) -> dict:
    h, w = map(int, g.shape)
    candidates: list[dict] = []

    for inner_size in range(MIN_INNER, MAX_INNER + 1):
        total_size = inner_size + 1
        for edge, r0, c0 in _iter_edge_windows(h, w, total_size):
            for orientation in EDGE_TO_ORIENTATIONS[edge]:
                border_cells, inside_slice = _legend_layout(r0, c0, inner_size, orientation)
                border_vals = {int(g[r, c]) for r, c in border_cells}
                if len(border_vals) != 1:
                    continue
                border_color = next(iter(border_vals))
                if border_color == int(bg):
                    continue
                if int(np.count_nonzero(g == border_color)) != len(border_cells):
                    continue

                inside = g[inside_slice]
                if np.any(inside == border_color):
                    continue
                colors = set(map(int, np.unique(inside)))
                colors.discard(int(bg))
                if len(colors) < 2:
                    continue

                candidates.append(
                    {
                        "edge": edge,
                        "orientation": orientation,
                        "r0": int(r0),
                        "c0": int(c0),
                        "inner_size": int(inner_size),
                        "total_size": int(total_size),
                        "border_color": int(border_color),
                        "colors": colors,
                    }
                )

    assert len(candidates) == 1, f"expected exactly one legend, found {len(candidates)}"
    return candidates[0]


def _legend_window_mask(shape: tuple[int, int], legend: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0 = int(legend["r0"])
    c0 = int(legend["c0"])
    sz = int(legend["total_size"])
    mask[r0 : r0 + sz, c0 : c0 + sz] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove ordinary objects that realize the full color set encoded by a movable edge legend.

    Input structure:
    - Background fills most of the board.
    - One compact square legend touches exactly one outer edge and avoids the corners.
    - The legend is marked by a single-color L border whose color is not fixed and does not
      appear in the ordinary objects; its interior side length can be 2, 3, 4, or 5.
    - Elsewhere are multicolor 4-connected components drawn on the background.

    Rule:
    - Locate the unique edge legend by finding an edge-attached L frame whose color appears
      only on that frame.
    - Read the target palette from the legend interior, ignoring background.
    - Ignore the legend window itself, then inspect every other 4-connected non-background
      component.
    - Erase each such component to background iff it contains every legend color.

    Key insight: the legend is no longer identified by a fixed corner, a fixed green border,
    or a fixed 3x3 interior; it must be recognized structurally as the only exclusive-color
    L frame on the boundary, after which the original "contains all legend colors" filter
    still decides which ordinary objects disappear.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend = _find_legend(g, bg)
    legend_colors = set(map(int, legend["colors"]))
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    legend_mask = _legend_window_mask(g.shape, legend)
    comps = components_mask((g != bg) & (~legend_mask), connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = int(bg)
    return out


def _legend_anchor(inner_size: int, edge: str) -> tuple[int, int]:
    sz = int(inner_size) + 1
    if edge == "top":
        return 0, random.randint(1, GRID_W - sz - 1)
    if edge == "bottom":
        return GRID_H - sz, random.randint(1, GRID_W - sz - 1)
    if edge == "left":
        return random.randint(1, GRID_H - sz - 1), 0
    if edge == "right":
        return random.randint(1, GRID_H - sz - 1), GRID_W - sz
    raise ValueError(f"unknown edge: {edge}")


def _place_legend(
    inp: np.ndarray,
    out: np.ndarray,
    targets: list[int],
    border_color: int,
    edge: str,
    orientation: str,
    inner_size: int,
) -> None:
    sz = int(inner_size) + 1
    patch = np.full((sz, sz), BG, dtype=int)
    border_cells, inside_slice = _legend_layout(0, 0, inner_size, orientation)
    for r, c in border_cells:
        patch[r, c] = int(border_color)

    cols = random.sample(range(inner_size), inner_size)
    for i, color in enumerate(targets):
        rr = int(inside_slice[0].start) + int(i)
        cc = int(inside_slice[1].start) + int(cols[i])
        patch[rr, cc] = int(color)

    r0, c0 = _legend_anchor(inner_size, edge)
    paste(inp, patch, r0, c0)
    paste(out, patch, r0, c0)


def _component_dims() -> tuple[int, int]:
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], palette: list[int], deleted: bool) -> np.ndarray:
    area = int(h * w)
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if deleted:
        lo = max(lo, len(targets))
        hi = max(hi, lo)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in palette if v not in targets]

    if deleted:
        colors = list(targets) + [random.choice(palette) for _ in range(n_colored - len(targets))]
    else:
        used_target_count = random.randint(0, len(targets) - 1)
        allowed_targets = random.sample(targets, used_target_count)
        allowed = non_targets + allowed_targets
        colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.full((h, w), BASE_COLOR, dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    edge = random.choice(EDGE_NAMES)
    inner_size = random.randint(MIN_INNER, MAX_INNER)
    border_color = random.choice(BORDER_POOL)
    palette = [c for c in BORDER_POOL if c != border_color]
    targets = random.sample(palette, inner_size)
    orientation = random.choice(EDGE_TO_ORIENTATIONS[edge])
    _place_legend(inp, out, targets, border_color, edge, orientation, inner_size)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, palette, deleted)
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if not deleted:
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
