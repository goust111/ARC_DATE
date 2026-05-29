# Combo variant of d59b0160 (index 32)
# combo_spec: D2/1+D14/1
# Name: 反转保留删除+输出提取被删除分量
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - The legend format stays the same: a 4x4 top-left region with border color 3 on the
#      bottom row and right column, and three target colors inside.
#    - Base task: a non-legend 4-connected component is deleted when it contains all three
#      legend colors; everything else stays in the full-size output.
#    - D2 flips that decision: matching components are the ones that would survive, while
#      incomplete components are the ones that would be deleted.
#    - D14 then extracts the deleted content instead of showing the whole scene. After the
#      flip, "deleted content" means the non-matching component set. Example:
#      legend={1,4,8}; component A has {0,1,4,8,5}; component B has {0,1,4}. The base task
#      erases A. The fused task keeps A conceptually, treats B as the deleted object, and
#      outputs only B's pre-deletion pattern cropped to its own bbox.
# b) Complexity calibration:
#    - If several non-matching components were extracted together, the crop could look like a
#      generic union-of-objects task rather than "the rejected object". The generator therefore
#      uses exactly one non-matching component and 3-5 matching components.
#    - That keeps the inversion visible: many objects satisfy the legend, but the output is the
#      lone object that fails the legend test.
# c) Implicit assumptions broken by the fusion:
#    - Scanning all non-bg components without filtering would now capture legend fragments,
#      because legend border/swatch components are also non-matching. The solver must skip
#      every component fully contained in the legend square.
#    - Returning a same-size canvas is no longer correct; after D14 the solver must crop to the
#      bbox of the deleted non-legend components and keep only those pixels inside the crop.
#    - The original code tolerated zero surviving matches. The fused rule is easier to infer if
#      matching components clearly exist, so the generator guarantees 3-5 of them.
# d) Placement legality:
#    - The generator still places each rectangle patch with
#      `random_free_location(..., transparent=7, padding=1, padding_connectivity=8)`.
#      This prevents overlap, accidental touching, and out-of-bounds placement, and it keeps
#      every non-legend component separated from the legend by at least one background cell.
# e) Visual observability:
#    - The output crop is visibly smaller than the 16x16 input, so extraction is obvious.
#    - The extracted object is also visibly missing at least one legend color, which separates
#      this fusion from the original D14-style "extract the matching component" behavior.
# f) Visual cleanliness:
#    - Only one rejected component is extractable, so the output is a single compact patch rather
#      than a noisy union crop. The input still contains several separated objects, preserving the
#      comparison structure needed for humans to see which one was rejected.
# g) Inferability of changes:
#    - No hidden color remapping is introduced. The same visible cue drives everything: whether an
#      object's palette fully covers the legend palette. The only twist is that the extracted
#      object is the one that fails that cue.
# h) Quantitative anti-degeneracy target:
#    - Accepted samples must have exactly one non-legend non-matching component, at least three
#      non-legend matching components, `main(input).shape != input.shape`, and the output palette
#      must miss at least one legend color.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes any component inside the legend square should be ignored; the generator
#      reserves the top-left 4x4 region for the legend only.
#    - `main()` assumes there is at least one deleted non-legend component to crop; the generator
#      always creates exactly one non-matching component.
#    - `main()` assumes the deleted component bbox is smaller than the whole grid; component size
#      ranges enforce that.

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

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the top-left legend whose bottom edge and right edge share one border color.
    Return (size, border_color, interior_target_colors).
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
            return int(sz), border_color, colors
    raise AssertionError("no legend border found in grid")


def _is_legend_component(pts: np.ndarray, sz: int) -> bool:
    """Legend fragments are confined to the top-left sz x sz square."""
    return bool(np.all(pts[:, 0] < sz) and np.all(pts[:, 1] < sz))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the non-matching component(s) that would be deleted after the keep/delete inversion.

    Input structure:
    - One background color fills the outer frame and the empty cells between objects.
    - The top-left corner contains a small legend box whose bottom row and right column share one
      border color, while the interior shows the target palette.
    - Away from that legend, several separated 4-connected multicolor components sit on the
      background.

    Rule:
    - Read the target color set from the legend interior.
    - Ignore any component contained entirely inside the legend square.
    - Among the remaining non-background components, mark as "deleted" every component whose color
      set does not contain all legend colors; matching components are the ones conceptually kept.
    - Output only the minimum crop covering the deleted component(s), preserving their original
      pixels and filling every other cell in the crop with the background color.

    Key insight: the legend still tests palette completeness, but the two variant changes interact:
    first the keep/delete decision is inverted, then the output extracts exactly the objects that the
    inverted rule deletes, i.e. the incomplete palette components.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    deleted: list[np.ndarray] = []
    for pts in components_mask(g != bg, connectivity=4):
        if _is_legend_component(pts, sz):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if not legend_colors.issubset(cols):
            deleted.append(pts)

    assert deleted, "no deleted non-legend component found"

    all_pts = np.concatenate(deleted, axis=0)
    r0, c0, r1, c1 = bbox_coords(all_pts)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for pts in deleted:
        for r, c in pts:
            out[int(r) - r0, int(c) - c0] = int(g[r, c])
    return out


# --------------- generator (combo 32) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [6, 2, 2]
TALL_H = (6, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 6)
MED_W = (3, 7)

DENSITY_LO = 0.10
DENSITY_HI = 0.50

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [3, 5, 4]
NUM_NONMATCH = 1

MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place the three target swatches."""
    s = LEGEND_SIZE
    inp[:s, :s] = BG
    inp[s - 1, :s] = BORDER
    inp[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        pos for pos in border_side
        if abs(pos[0] - isolated[0]) + abs(pos[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color


def _component_dims() -> tuple[int, int]:
    """Sample one rectangle size according to the shape weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], matches: bool) -> np.ndarray:
    """Build one solid non-bg rectangle with base color 0 and sprinkled visible colors."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if matches:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [color for color in PALETTE if color not in targets]

    if matches:
        colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - NUM_TARGETS)]
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed = non_targets + random.sample(targets, used_target_count)
        colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output is the unique deleted non-match crop."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [False] * NUM_NONMATCH + [True] * (component_count - NUM_NONMATCH)
    random.shuffle(roles)

    expected_out = None
    for matches in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, matches)
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
        if not matches:
            expected_out = patch.copy()

    if expected_out is None:
        return None
    if np.array_equal(main(inp), expected_out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
