# Combo variant of d59b0160 (index 29)
# combo_spec: D1/1+D5/2+D16/1
# Name: 精确颜色匹配+有序序列+空间模式匹配

import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior trace:
#    - The legend is still a 4x4 top-left box with border color 3, but its three colored interior
#      pixels now encode two things at once:
#        1. a row-major color sequence, and
#        2. the row-major spatial pattern formed by those three pixels.
#    - Each non-legend object is still one 4-connected component because its rectangle is filled
#      with color 0 and the colored markers are embedded inside that body.
#    - Original behavior on legend colors {2,5,8}:
#        component palette {0,2,5,8,6} -> deleted because it contains all legend colors.
#      New behavior:
#        exact-set branch rejects it immediately because the visible palette is {2,5,8,6}, not
#        exactly {2,5,8}.
#      Another object with visible palette {2,5,8} but colors placed in the same triangle shape
#      as the legend in order [5,2,8] is also kept: the set is exact and the shape matches, but
#      the ordered sequence does not.
#      A third object with visible palette {2,5,8} and sequence [2,5,8] can still survive if its
#      three target pixels form a different triangle than the legend.
#    - Therefore deletion now requires all three visible tests to pass simultaneously.
# b) Complexity calibration:
#    - The fusion is intentionally strict, so the generator uses only a small repeated family of
#      3-pixel templates. This keeps the spatial hypothesis learnable instead of exploding into
#      arbitrary point clouds.
#    - Every sample contains three informative kept roles:
#        superset       : order + pattern match, exact-set fails
#        permute        : exact-set + pattern match, order fails
#        shape_mismatch : exact-set + order match, pattern fails
#      This makes each constraint observable from a few examples instead of hiding the combo behind
#      a single monolithic "doesn't disappear" case.
# c) Implicit assumption audit:
#    - `_find_legend(...)` can no longer return only a set; it must preserve row-major order and
#      the row-major template coordinates.
#    - Comparing a component's raw color set would be wrong because the rectangle body color 0 is
#      always present, so matching must ignore bg and 0.
#    - Pattern matching becomes ambiguous if a target color appears multiple times. The new solver
#      therefore treats "exactly one pixel per target color" as part of the strict match, and the
#      generator enforces that for all informative components.
#    - Sequence and pattern are kept separable by defining the pattern on the row-major slots
#      themselves, not on color ids. That allows "same shape, wrong color order" examples.
# d) Placement legality:
#    - Components are generated as compact rectangles (at least 3x3 so the 3-point templates fit)
#      and placed with `random_free_location(..., border=1, padding=1)`, so they never overlap,
#      never touch, stay away from the frame, and remain separated from the legend.
#    - Template offsets are translated inside each rectangle only after checking they fit.
# e) Visual observability:
#    - A human can compare three full-palette objects: one disappears, one keeps the same shape but
#      swaps the color order, and one keeps the same color order but changes the triangle/line
#      pattern. That directly exposes the fused rule.
#    - No hidden numeric ordering is used; both order and pattern come from visible row-major
#      reading of pixels.
# f) Visual cleanliness:
#    - Objects remain isolated rectangles with only 3 or 4 colored markers, so the evidence is
#      sparse and easy to inspect rather than buried in noisy textures.
#    - The legend stays in a dedicated corner region and reuses the same small template family as
#      the objects, which keeps the scene organized.
# g) Change inferability:
#    - The palette stays small and only three legend colors matter at a time.
#    - Spatial matching is supported by repeated template reuse, not by arbitrary one-off geometry.
# 9) New solver assumptions checked against the generator:
#    - main() assumes the legend contains exactly three distinct non-bg/non-border colors; the
#      generator always places exactly one copy of each target color in the legend interior.
#    - main() assumes deletable objects have exactly one pixel of each target color and no extra
#      visible colors; the generator constructs `delete` objects that way.
#    - main() assumes there is at least one full-palette kept object that differs from the delete
#      case by one of the three constraints; the generator always includes `superset`, `permute`,
#      and `shape_mismatch` roles.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _normalize_pattern(positions: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Normalize row-major positions by translating the first point to the origin."""
    r0, c0 = positions[0]
    return tuple((int(r - r0), int(c - c0)) for r, c in positions)


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, list[int], tuple[tuple[int, int], ...]]:
    """
    Detect the top-left legend and read both its ordered colors and its relative 3-point pattern.

    The legend sequence is the row-major scan of the legend interior, skipping bg and border cells.
    The legend pattern is the row-major list of those same positions, normalized by translation.
    """
    h, w = g.shape
    sizes = [LEGEND_SIZE] + [sz for sz in range(2, min(h, w) + 1) if sz != LEGEND_SIZE]
    for sz in sizes:
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            seq: list[int] = []
            pos: list[tuple[int, int]] = []
            for r in range(inside.shape[0]):
                for c in range(inside.shape[1]):
                    color = int(inside[r, c])
                    if color not in (bg, x):
                        seq.append(color)
                        pos.append((int(r), int(c)))
            return int(sz), int(x), seq, _normalize_pattern(pos)
    assert False, "no legend border found in grid"


def _component_visible_palette(g: np.ndarray, pts: np.ndarray, bg: int) -> set[int]:
    """Visible object palette for matching: discard global bg and the internal filler 0."""
    cols = {int(g[r, c]) for r, c in pts}
    cols.discard(bg)
    cols.discard(0)
    return cols


def _component_sequence_and_pattern(
    g: np.ndarray,
    pts: np.ndarray,
    targets: set[int],
) -> tuple[list[int], tuple[tuple[int, int], ...]] | None:
    """
    Return the component's target-color row-major sequence and target-pixel pattern.

    The strict combo match only accepts components with exactly one pixel for each target color.
    """
    r0, c0 = pts.min(axis=0)
    entries: list[tuple[int, int, int]] = []
    counts = {color: 0 for color in targets}
    for r, c in pts:
        color = int(g[r, c])
        if color in targets:
            rr = int(r - r0)
            cc = int(c - c0)
            entries.append((rr, cc, color))
            counts[color] += 1

    if any(counts[color] != 1 for color in targets):
        return None

    entries.sort()
    seq = [color for _rr, _cc, color in entries]
    positions = [(rr, cc) for rr, cc, _color in entries]
    return seq, _normalize_pattern(positions)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete only those components that exactly replicate the legend's colors, order, and pattern.

    Input structure:
    - The border-defined background still fills most of the grid.
    - The top-left 4x4 legend keeps the same bottom/right frame, but its three interior colors now
      encode an ordered sequence together with a 3-point spatial template.
    - Elsewhere, each object is a 4-connected rectangle of color 0 containing a few colored markers.

    Rule:
    - Read the legend by scanning its 3x3 interior in row-major order, keeping the three non-bg,
      non-border colors as an ordered sequence and recording their normalized row-major positions.
    - For each connected component, ignore bg and 0 when forming its visible palette.
    - Erase the component iff all three constraints hold at once:
        1. its visible palette is exactly the legend color set,
        2. each target color appears exactly once and those three pixels read in the legend order,
        3. those same three pixels form the same normalized row-major position pattern as the legend.
    - Keep every component that breaks any one of the three constraints.

    Key insight: The legend is no longer just a color filter; it is a full three-part signature:
    exact palette, ordered color assignment, and matching local geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_seq, legend_pattern = _find_legend(g, bg)
    assert len(legend_seq) == NUM_TARGETS, "legend must contain exactly three target colors"
    target_set = set(legend_seq)
    assert len(target_set) == NUM_TARGETS, "legend colors must be distinct"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if _component_visible_palette(g, pts, bg) != target_set:
            continue
        seq_and_pattern = _component_sequence_and_pattern(g, pts, target_set)
        if seq_and_pattern is None:
            continue
        seq, pattern = seq_and_pattern
        if seq == legend_seq and pattern == legend_pattern:
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (combo 29) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

LEGEND_TEMPLATES = [
    [(0, 0), (1, 2), (2, 1)],
    [(0, 1), (1, 0), (2, 2)],
    [(0, 2), (1, 0), (2, 1)],
]

COMPONENT_SIZES = [
    (3, 3), (3, 4), (3, 5),
    (4, 3), (4, 4), (4, 5),
    (5, 3), (5, 4), (5, 5),
]
COMPONENT_WEIGHTS = [2, 2, 1, 2, 4, 2, 1, 2, 1]

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [2, 5, 10]

MAX_RETRIES = 500


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> list[tuple[int, int]]:
    """Draw the 4x4 legend box and place the ordered target sequence in one template."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    positions = random.choice(LEGEND_TEMPLATES)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color
    return positions


def _component_dims() -> tuple[int, int]:
    """Sample one compact rectangle size."""
    return random.choices(COMPONENT_SIZES, weights=COMPONENT_WEIGHTS)[0]


def _shifted_pattern(h: int, w: int, pattern: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Translate a 3-point pattern so it fits inside an h x w component."""
    max_r = max(r for r, _c in pattern)
    max_c = max(c for _r, c in pattern)
    off_r = random.randint(0, h - 1 - max_r)
    off_c = random.randint(0, w - 1 - max_c)
    return [(int(r + off_r), int(c + off_c)) for r, c in pattern]


def _wrong_permutation(targets: list[int]) -> list[int]:
    """Random non-identity permutation of the legend sequence."""
    perms = [
        [targets[0], targets[2], targets[1]],
        [targets[1], targets[0], targets[2]],
        [targets[1], targets[2], targets[0]],
        [targets[2], targets[0], targets[1]],
        [targets[2], targets[1], targets[0]],
    ]
    return random.choice(perms)


def _extra_coord(h: int, w: int, used: set[tuple[int, int]]) -> tuple[int, int]:
    """Pick one free coordinate for the superset role."""
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in used]
    return random.choice(free)


def _component_patch(
    h: int,
    w: int,
    targets: list[int],
    legend_pattern: list[tuple[int, int]],
    role: str,
) -> np.ndarray:
    """
    Build one component patch with a 0-colored body and a role-specific marker layout.

    Roles:
      - delete: exact palette, legend sequence, legend pattern
      - superset: delete pattern plus one non-target color
      - permute: exact palette, legend pattern, wrong color order
      - shape_mismatch: exact palette, legend sequence, wrong spatial pattern
    """
    patch = np.zeros((h, w), dtype=int)
    non_targets = [v for v in PALETTE if v not in targets]

    if role == "shape_mismatch":
        other_patterns = [p for p in LEGEND_TEMPLATES if p != legend_pattern]
        anchors = _shifted_pattern(h, w, random.choice(other_patterns))
        colors = list(targets)
    else:
        anchors = _shifted_pattern(h, w, legend_pattern)
        if role == "permute":
            colors = _wrong_permutation(targets)
        else:
            colors = list(targets)

    used: set[tuple[int, int]] = set()
    for color, (r, c) in zip(colors, anchors):
        patch[r, c] = color
        used.add((r, c))

    if role == "superset":
        rr, cc = _extra_coord(h, w, used)
        patch[rr, cc] = random.choice(non_targets)

    return patch


def _component_roles() -> list[str]:
    """Choose 1-3 deleted objects plus three informative kept objects."""
    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - 3
    roles = ["delete"] * deleted_count + ["superset", "permute", "shape_mismatch"]
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose intended output matches `main`."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    legend_pattern = _place_legend(inp, out, targets)

    for role in _component_roles():
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, legend_pattern, role)
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
        if role != "delete":
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
