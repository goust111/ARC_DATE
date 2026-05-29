# Combo variant of d59b0160 (index 6)
# combo_spec: D3/2+D5/0
# Name: 分散像素标记图例+特定位置提取颜色
#
# Design analysis
# ===============
# a) Concrete behavior and execution trace:
#    - Preserve the original component rule: non-background 4-connected rectangles are erased iff
#      their color set contains every target color.
#    - Replace the old top-left boxed legend with four distributed green border markers, one per edge.
#      Each marker owns a 3-cell inward strip:
#        marker(3) -> near colored cell -> far colored cell
#    - Only the far cell counts toward the target set. The near cell is a distractor.
#    - Small trace:
#      top strip:    3 -> 5 -> 1
#      bottom strip: 3 -> 6 -> 4
#      left strip:   3 -> 2 -> 8
#      right strip:  3 -> 5 -> 1
#      Fusion rule reads far colors {1,4,8}. A component with {0,1,4,8,9} is deleted.
#      A component with {0,2,5,6,9} stays even though it contains all near colors {2,5,6}.
#      Under the D3-only reading ("use the adjacent swatch"), the second component would be deleted
#      and the first one could be kept if it misses a near color. That is the intended divergence.
#    - Relative to the source task, two steps changed together:
#      1. legend discovery moved from one compact corner box to four globally scattered edge controls;
#      2. legend lookup no longer means "every shown color matters", because only one fixed slot in
#         each local strip is meaningful.
# b) Complexity calibration:
#    - Keep exactly four markers and exactly one strip per edge. This is enough repetition for humans
#      to notice the border-control grammar without exploding the search space.
#    - Keep exactly three distinct target colors, with one repeated across the four far cells.
#      The near distractor colors follow the same 3-distinct-plus-repeat pattern so the solver must
#      form a set rather than memorize one edge.
#    - Use disjoint target and distractor palettes. This makes the "wrong position" interpretation
#      visibly fail instead of accidentally matching the correct output.
# c) Implicit-assumption audit for the original main():
#    - `_find_legend(...)` from the source solver is invalid because there is no top-left frame now.
#    - `_bg_boundary(...)` still holds because the border remains mostly background apart from four
#      isolated green markers.
#    - The component sweep over `g != bg` still works, but now the legend contributes four tiny
#      components. Each such strip contains only one far target, so none can satisfy the full target set.
#    - The new legend reader assumes each green border pixel is not on a corner and has a 2-cell inward
#      colored strip. The generator enforces this by fixed edge placement and padding.
# d) Placement legality:
#    - Markers avoid corners, and each strip stays fully inside the 16x16 grid by using a 2-cell inward ray.
#    - Components are placed only after the border controls are drawn, with `padding=1`, so they cannot
#      overlap or touch any strip in the 8-neighborhood.
#    - The top-left 4x4 region is kept empty to avoid accidental resemblance to the original framed legend.
# e) Visual observability:
#    - Across examples, green singletons on the border are the only obvious legend cue, so humans can
#      spot that the legend is distributed.
#    - Each cue has two colored cells, and training pairs can show that deleting by the near colors is
#      wrong while deleting by the far colors is right. This makes the position-sensitive lookup visible.
# f) Visual cleanliness:
#    - One strip per edge keeps the control layer sparse and separated from the object field.
#    - Object rectangles remain padded apart, so the grid does not become cluttered even with extra decoys.
# g) Change inferability and anti-degeneration guard:
#    - All deleted objects contain every far target but miss at least one near distractor.
#    - At least one kept object contains every near distractor but misses a far target.
#    - Therefore the fusion logic and the simpler D3-only logic necessarily produce different outputs on
#      generated samples; the generator rejects any accidental degeneration where they match anyway.

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _edge_dir(side: str) -> tuple[int, int]:
    if side == "top":
        return 1, 0
    if side == "bottom":
        return -1, 0
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, -1
    raise ValueError(f"unknown side: {side}")


def _marker_records(g: np.ndarray, bg: int) -> dict[str, dict[str, object]]:
    """
    Extract one distributed legend strip per edge.

    Each record stores:
      - marker: green border cell
      - near: adjacent inward distractor cell
      - far: second inward target cell
    """
    h, w = g.shape
    records: dict[str, dict[str, object]] = {}
    for r, c in np.argwhere(g == BORDER):
        r = int(r)
        c = int(c)
        if r == 0 and 0 < c < w - 1:
            side = "top"
        elif r == h - 1 and 0 < c < w - 1:
            side = "bottom"
        elif c == 0 and 0 < r < h - 1:
            side = "left"
        elif c == w - 1 and 0 < r < h - 1:
            side = "right"
        else:
            continue

        dr, dc = _edge_dir(side)
        near = (r + dr, c + dc)
        far = (r + 2 * dr, c + 2 * dc)
        nr, nc = near
        fr, fc = far
        assert 0 <= fr < h and 0 <= fc < w, "marker strip exits the grid"

        near_color = int(g[nr, nc])
        far_color = int(g[fr, fc])
        assert near_color not in (bg, BORDER), "near strip cell must be a distractor color"
        assert far_color not in (bg, BORDER), "far strip cell must be a target color"

        records[side] = {
            "marker": (r, c),
            "near": near,
            "far": far,
            "near_color": near_color,
            "far_color": far_color,
        }

    assert set(records) == set(EDGE_ORDER), "expected one distributed legend strip on each edge"
    return records


def _legend_target_colors(g: np.ndarray, bg: int) -> set[int]:
    """Read only the far position of each distributed edge strip."""
    return {int(rec["far_color"]) for rec in _marker_records(g, bg).values()}


def _legend_distractor_colors(g: np.ndarray, bg: int) -> set[int]:
    """Read the adjacent strip colors; used for anti-degeneration checks."""
    return {int(rec["near_color"]) for rec in _marker_records(g, bg).values()}


def _erase_matching_components(g: np.ndarray, bg: int, required_colors: set[int]) -> np.ndarray:
    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if required_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose colors match the far slot of distributed edge strips.

    Input structure:
    - Background fills most of the 16x16 grid.
    - On each outer edge there is one green single-pixel marker, and immediately inside it sits a
      2-cell colored strip running perpendicular to the border.
    - The first colored cell in each strip is a distractor; the second colored cell is the meaningful swatch.
    - Elsewhere, multicolored 4-connected rectangle components occupy the background.

    Rule:
    - Scan the whole border to find the four green markers.
    - For each marker, look two cells inward and collect that far color into the target set.
    - For every non-background connected component, test whether its color set contains all target colors.
    - Erase components that do, and keep everything else unchanged.

    Key insight: The legend must be decoded in two spatial stages: first gather the distributed border
    clues, then read the correct slot inside each local strip instead of trusting the adjacent color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_colors = _legend_target_colors(g, bg)
    assert legend_colors, "distributed legend contains no target colors"
    return _erase_matching_components(g, bg, legend_colors)


# --------------- generator (combo variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3
NUM_DISTRACTORS = 3

SHAPE_WEIGHTS = [7, 2, 1]
TALL_H = (6, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (4, 8)
MED_H = (3, 7)
MED_W = (2, 9)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [8, 4, 1]
NUM_KEPT = 3

MAX_RETRIES = 1200
EDGE_ORDER = ("top", "bottom", "left", "right")


def _edge_strip(edge: str, pos: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if edge == "top":
        return (0, pos), (1, pos), (2, pos)
    if edge == "bottom":
        return (GRID_H - 1, pos), (GRID_H - 2, pos), (GRID_H - 3, pos)
    if edge == "left":
        return (pos, 0), (pos, 1), (pos, 2)
    if edge == "right":
        return (pos, GRID_W - 1), (pos, GRID_W - 2), (pos, GRID_W - 3)
    raise ValueError(f"unknown edge: {edge}")


def _place_distributed_legend(
    inp: np.ndarray,
    out: np.ndarray,
    targets: list[int],
    distractors: list[int],
) -> None:
    """Place four green edge markers, each with a distractor cell and a farther target cell."""
    target_sequence = list(targets) + [random.choice(targets)]
    distractor_sequence = list(distractors) + [random.choice(distractors)]
    random.shuffle(target_sequence)
    random.shuffle(distractor_sequence)

    positions = {
        "top": random.randint(5, GRID_W - 3),
        "bottom": random.randint(2, GRID_W - 3),
        "left": random.randint(5, GRID_H - 3),
        "right": random.randint(2, GRID_H - 3),
    }

    for edge, near_color, far_color in zip(EDGE_ORDER, distractor_sequence, target_sequence):
        marker, near, far = _edge_strip(edge, positions[edge])
        for g in (inp, out):
            g[marker] = BORDER
            g[near] = near_color
            g[far] = far_color


def _component_dims() -> tuple[int, int]:
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, mandatory: list[int], allowed: list[int]) -> np.ndarray:
    """Build one rectangle component whose sprinkled palette obeys the requested color constraints."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO), len(mandatory))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    pool = list(allowed) if allowed else list(mandatory)
    assert pool, "component color pool cannot be empty"

    colors = list(mandatory) + [random.choice(pool) for _ in range(n_colored - len(mandatory))]
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _deleted_patch(h: int, w: int, targets: list[int], distractors: list[int], non_special: list[int]) -> np.ndarray:
    """
    Deleted objects always contain every far target but miss at least one near distractor.

    This makes the combo behavior diverge from the simpler "use the adjacent swatches" interpretation.
    """
    used_distractor_count = random.randint(0, NUM_DISTRACTORS - 1)
    used_distractors = random.sample(distractors, used_distractor_count)
    mandatory = list(targets) + used_distractors
    allowed = list(targets) + used_distractors + non_special
    return _component_patch(h, w, mandatory, allowed)


def _kept_patch(
    h: int,
    w: int,
    targets: list[int],
    distractors: list[int],
    non_special: list[int],
    require_all_distractors: bool,
) -> np.ndarray:
    """Kept objects miss at least one far target; one witness object carries all near distractors."""
    used_target_count = random.randint(0, NUM_TARGETS - 1)
    used_targets = random.sample(targets, used_target_count)
    if require_all_distractors:
        used_distractors = list(distractors)
    else:
        used_distractor_count = random.randint(0, NUM_DISTRACTORS)
        used_distractors = random.sample(distractors, used_distractor_count)
    mandatory = used_targets + used_distractors
    allowed = used_targets + used_distractors + non_special
    if not allowed:
        allowed = list(non_special)
    return _component_patch(h, w, mandatory, allowed)


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    chosen = random.sample(PALETTE, NUM_TARGETS + NUM_DISTRACTORS)
    targets = chosen[:NUM_TARGETS]
    distractors = chosen[NUM_TARGETS:]
    non_special = [v for v in PALETTE if v not in targets and v not in distractors]
    _place_distributed_legend(inp, out, targets, distractors)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    roles = ["deleted"] * deleted_count + ["kept"] * NUM_KEPT
    random.shuffle(roles)

    kept_with_all_distractors = False
    for role in roles:
        h, w = _component_dims()
        if role == "deleted":
            patch = _deleted_patch(h, w, targets, distractors, non_special)
        else:
            require_all_distractors = not kept_with_all_distractors
            patch = _kept_patch(h, w, targets, distractors, non_special, require_all_distractors)
            kept_with_all_distractors |= require_all_distractors

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
        if role == "kept":
            paste(out, patch, r, c)

    if not kept_with_all_distractors:
        return None
    if np.any(inp[:4, :4] != BG):
        return None

    bg = _bg_boundary(inp)
    if np.array_equal(_erase_matching_components(inp, bg, _legend_distractor_colors(inp, bg)), out):
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
