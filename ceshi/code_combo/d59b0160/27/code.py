# Combo variant of d59b0160 (index 27)
# combo_spec: D2/0+D6/3+D13/0
# Name: 匹配分量重着色+幽灵框架+保留分量统一色

"""
Design analysis for the combo variant
=====================================

a) Concretization and behavior trace
------------------------------------
- The legend is still the same 4x4 top-left box: bottom row and right column are border color 3,
  and the interior shows exactly three target colors.
- Every ordinary object is now generated as one 4-connected irregular shape inside its own bbox,
  not as a full solid rectangle. This matters because the D6 ghost-frame layer must remain visible.
- Example:
  - legend colors = {1, 4, 8}
  - component A colors = {0, 1, 4, 8, 9}; A touches all four sides of its bbox but leaves many
    bbox-border cells empty.
  - component B colors = {0, 6, 9}; B misses the legend colors and has dominant color 0.
  - original task:
    - A matches, so A is erased to background.
    - B does not match, so B is copied unchanged.
  - combo task:
    - A matches, so A's own occupied cells become border color 3, and the entire bbox border is
      also drawn in color 3. The result is a green silhouette plus a green rectangular frame.
    - B does not match, so B keeps only its shape and becomes a flat block of its most common
      input color, here 0.
  - Difference from the original: both branches now visibly transform the object, and the matched
    branch has two simultaneous effects rather than simple removal.

b) Complexity calibration
-------------------------
- If objects stayed as full rectangles, "recolor to 3" plus "draw bbox frame in 3" would collapse
  into one effect; the ghost-frame dimension would become invisible. Therefore the generator uses
  irregular connected masks that touch all four bbox sides but do not fill the bbox.
- The replacement colors are kept simple and repeated:
  - matched objects always go to the fixed border color 3,
  - kept objects always go to their own unique majority color.
- The generator still produces 1-3 matched objects and exactly 3 kept objects so both branches are
  consistently represented.

c) Implicit assumption audit
----------------------------
- `_bg_boundary` still assumes the outer frame is mostly background 7. The generator keeps the
  perimeter background-dominated.
- `_find_legend` still assumes the smallest qualifying top-left square is the legend. The legend
  geometry is unchanged, so this remains valid.
- `components_mask(g != bg, connectivity=4)` now depends on each object being a single connected
  non-background shape even though it may have background holes around it inside its reserved bbox.
  The generator explicitly builds connected masks.
- The original solver could safely iterate over all components because unchanged non-legend objects
  were copied as-is. In this combo, recoloring every non-matching component would also recolor the
  legend pieces, so `main()` must skip components fully contained in the legend square.
- The new D13 branch assumes the kept component has a unique mode color. The generator guarantees a
  strict majority by constructing every object from one dominant color plus minority mutations.
- The new D6 branch assumes the bbox frame can be drawn without colliding with neighbors. The
  generator reserves the full bbox during placement and still leaves a 1-cell padded gap.

d) Placement legality
---------------------
- Each component is placed by reserving its full bbox with `random_free_location(..., padding=1)`.
  This blocks overlaps both in the input shape itself and in the output ghost frame drawn later.
- Patches are pasted with `transparent=BG`, so background holes inside the bbox stay empty while the
  placement logic still treats the entire bbox as occupied.
- All drawing stays in-bounds because the frame is drawn only on the component's own bbox.

e) Visual observability
-----------------------
- A human can compare 2-4 examples and see two distinct output signatures:
  - matched objects always become green and acquire extra rectangular scaffolding,
  - kept objects lose internal texture and collapse to one dominant color.
- No invisible sorting or numeric color ordering is used. The two recolor targets are justified by
  visible cues: the legend border for matched objects, and the component's own dominant color for
  kept objects.

f) Visual cleanliness
---------------------
- The output uses only one overlay per matched object: a bbox frame drawn in the same salient color
  as the recolored shape. Because bboxes are reserved and separated, these frames do not overlap.
- Kept objects are simplified rather than decorated, which balances the matched-branch complexity.

g) Change inferability
----------------------
- The color mapping is intentionally low-entropy:
  - one global color (legend border) for all matched objects,
  - one local majority color per kept object.
- The generator forces every kept object to contain at least one minority color, so the flattening
  is visible, and forces every matched object to leave missing border cells, so the ghost frame is
  visible. This prevents degeneration back into the original task or into a single-dimension change.

Post-change assumption review
-----------------------------
- `main()` now assumes:
  - legend components can be recognized by full containment inside the legend square,
  - kept components have a meaningful majority color,
  - matched components may be non-rectangular, but their bbox frame is still interpretable.
- `generate_input()` enforces all three assumptions directly.
"""

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the smallest top-left square whose bottom row and right column share one border color.
    Return (size, border_color, set_of_target_colors_inside).
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


def _mode_color(values: np.ndarray) -> int:
    """Return the most frequent color; ties break by np.unique ordering."""
    vals, cnts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def _draw_bbox_border(out: np.ndarray, pts: np.ndarray, color: int) -> None:
    """Draw the inclusive bounding-box border of `pts` into `out`."""
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform legend-matching components into green silhouettes with ghost frames, and flatten the rest.

    Input structure:
    - Background fills most of the board.
    - A top-left legend square has one shared border color on its bottom edge and right edge; the
      interior cells list the target palette.
    - Outside that legend, each object is one 4-connected non-background component, possibly
      irregularly shaped but separated from the others by background.

    Rule:
    - Read the target color set from the legend interior and the frame color from the legend border.
    - For each non-legend connected component, test whether it contains all target colors.
    - Matching components are recolored on their own occupied cells to the legend-border color, and
      their entire bounding-rectangle border is also drawn in that same color.
    - Non-matching components keep their shape but are repainted uniformly to their own most common
      input color.

    Key insight:
    - The legend still chooses which objects are "selected", but now selected objects become marked
      green landmarks with rectangular scaffolds, while unselected objects are simplified into flat
      monochrome silhouettes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        rows, cols = pts[:, 0], pts[:, 1]
        if bool(np.all(rows < sz) and np.all(cols < sz)):
            continue

        comp_vals = g[rows, cols]
        comp_colors = set(map(int, comp_vals))
        if legend_colors.issubset(comp_colors):
            out[rows, cols] = border_color
            _draw_bbox_border(out, pts, border_color)
        else:
            out[rows, cols] = _mode_color(comp_vals)
    return out


# --------------- generator (combo variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
OBJECT_COLORS = [0] + PALETTE
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (5, 9)
TALL_W = (3, 4)
WIDE_H = (3, 4)
WIDE_W = (5, 8)
MED_H = (4, 6)
MED_W = (4, 6)

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [3, 6, 5]
NUM_KEPT = 3

MATCH_FILL = (0.35, 0.65)
KEEP_FILL = (0.45, 0.75)
MIN_MATCH_FRAME_GAIN = 4

MAX_RETRIES = 500


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend at top-left and place the 3 target colors."""
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
    """Sample component bbox dimensions; matched objects get slightly roomier boxes."""
    min_h = 4 if matched else 3
    min_w = 4 if matched else 3
    min_area = 12 if matched else 9

    while True:
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h = random.randint(*TALL_H)
            w = random.randint(*TALL_W)
        elif mode == "wide":
            h = random.randint(*WIDE_H)
            w = random.randint(*WIDE_W)
        else:
            h = random.randint(*MED_H)
            w = random.randint(*MED_W)

        h = max(h, min_h)
        w = max(w, min_w)
        if h * w >= min_area:
            return h, w


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Randomized Manhattan path from a to b, inclusive."""
    r, c = a
    path = [(r, c)]
    while (r, c) != b:
        steps: list[tuple[int, int]] = []
        if r < b[0]:
            steps.append((r + 1, c))
        elif r > b[0]:
            steps.append((r - 1, c))
        if c < b[1]:
            steps.append((r, c + 1))
        elif c > b[1]:
            steps.append((r, c - 1))
        r, c = random.choice(steps)
        path.append((r, c))
    return path


def _edge_anchors(h: int, w: int) -> list[tuple[int, int]]:
    """Choose one anchor on each side so the final shape spans the full bbox."""
    return [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]


def _frame_gain(mask: np.ndarray) -> int:
    """How many bbox-border cells are empty and would become visible only via the ghost frame."""
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return int(np.count_nonzero(border & ~mask))


def _build_component_mask(h: int, w: int, matched: bool) -> np.ndarray:
    """Create one 4-connected shape that touches all four sides of its bbox without filling it."""
    lo_ratio, hi_ratio = MATCH_FILL if matched else KEEP_FILL
    min_cells = 8 if matched else 5
    area = h * w

    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        center = (random.randint(1, h - 2), random.randint(1, w - 2))
        anchors = _edge_anchors(h, w)
        random.shuffle(anchors)

        for anchor in anchors:
            for r, c in _random_path(center, anchor):
                mask[r, c] = True

        skeleton_area = int(mask.sum())
        target_lo = max(min_cells, skeleton_area, math.ceil(area * lo_ratio))
        target_hi = min(area - 1, math.floor(area * hi_ratio))
        if target_hi < target_lo:
            if skeleton_area >= area:
                continue
            target = skeleton_area
        else:
            target = random.randint(target_lo, target_hi)

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            occupied = np.argwhere(mask)
            for r, c in occupied:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.add((int(rr), int(cc)))
            if not frontier:
                break
            rr, cc = random.choice(list(frontier))
            mask[rr, cc] = True

        if matched and _frame_gain(mask) < MIN_MATCH_FRAME_GAIN:
            continue
        return mask

    raise ValueError("failed to build visible connected component mask")


def _color_component(mask: np.ndarray, targets: list[int], matched: bool) -> tuple[np.ndarray, int]:
    """
    Color one connected mask with a strict dominant color.

    For matched components, all legend colors appear.
    For kept components, at least one legend color is absent.
    """
    coords = [tuple(rc) for rc in np.argwhere(mask)]
    n_cells = len(coords)
    patch = np.full(mask.shape, BG, dtype=int)
    non_targets = [v for v in OBJECT_COLORS if v not in targets]

    if matched:
        allowed = list(OBJECT_COLORS)
        dominant = random.choice(allowed)
        required = [t for t in targets if t != dominant]
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed_targets = random.sample(targets, used_target_count)
        allowed = non_targets + allowed_targets
        dominant = random.choice(allowed)
        required = [random.choice([v for v in allowed if v != dominant])]

    alternatives = [v for v in allowed if v != dominant]
    if not alternatives:
        raise ValueError("component needs at least one minority color")

    max_mutations = (n_cells - 1) // 2
    min_mutations = len(required)
    if min_mutations > max_mutations:
        raise ValueError("component too small for a strict dominant color")
    n_mutations = random.randint(min_mutations, max_mutations)

    for r, c in coords:
        patch[r, c] = dominant

    mutation_coords = random.sample(coords, n_mutations)
    mutation_colors = list(required)
    mutation_colors.extend(random.choice(alternatives) for _ in range(n_mutations - len(required)))
    random.shuffle(mutation_colors)

    for (r, c), color in zip(mutation_coords, mutation_colors):
        patch[r, c] = color
    return patch, int(dominant)


def _matched_output_patch(mask: np.ndarray) -> np.ndarray:
    """Render the matched-object output: green occupied cells plus a green bbox frame."""
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = BORDER
    patch[0, :] = BORDER
    patch[-1, :] = BORDER
    patch[:, 0] = BORDER
    patch[:, -1] = BORDER
    return patch


def _kept_output_patch(mask: np.ndarray, dominant: int) -> np.ndarray:
    """Render the kept-object output: same shape, flattened to one color."""
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(dominant)
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid combo sample; return the input grid on success."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for matched in roles:
        try:
            h, w = _component_dims(matched)
            mask = _build_component_mask(h, w, matched)
            patch, dominant = _color_component(mask, targets, matched)
            blocker = np.zeros((h, w), dtype=int)
            r, c = random_free_location(
                inp,
                blocker,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c, transparent=BG)
        if matched:
            paste(out, _matched_output_patch(mask), r, c, transparent=BG)
        else:
            paste(out, _kept_output_patch(mask, dominant), r, c, transparent=BG)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
