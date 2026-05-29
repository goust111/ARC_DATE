# Combo variant of d59b0160 (index 13)
# combo_spec: D9/1+D14/2
# Name: 矩形区域作为分量+多区域分别输出

import random

import numpy as np

from arc_utils import components_mask, paste

# Design analysis for the combo variant
# a) Concrete behavior:
#    The source task reads three target colors from the 4x4 top-left legend and
#    removes any non-bg connected component that contains all three colors,
#    leaving the answer on the original 16x16 canvas. That would collapse with
#    D9 if we kept the old generator's solid rectangles, because a solid
#    rectangle is already one 4-connected component. To make "rectangle as unit"
#    real and visible, each object here is instead a black rectangular frame
#    containing colored marks on the orange background. Internal marks may be
#    disconnected from the frame and from one another. The fused rule is:
#    identify each frame's full bounding box as one object, test the union of
#    colors inside that box against the legend, discard whole boxes that contain
#    all legend colors, and horizontally concatenate the surviving box crops.
#    Example trace:
#    - Box A: black frame + blue pixel + red pixel, but missing yellow -> kept.
#    - Box B: black frame + blue/red/yellow cluster -> deleted as one box.
#    - Under the old free-component logic, Box B would split into a frame
#      component and an interior cluster; under the fused logic the whole frame
#      region is the analysis unit, so the entire rectangle disappears.
# b) Complexity calibration:
#    Only one new grouping cue is introduced: the black rectangle outline. The
#    output layout also uses a single fixed rule, left-to-right concatenation
#    with a 1-cell bg gap. There is no branching between horizontal and vertical
#    packing, and no hidden tie-break based on color ids.
# c) Implicit assumptions from the old solver:
#    - `components_mask(g != bg)` is no longer valid, because internal
#      disconnected marks inside the same frame must stay grouped together.
#    - The legend must be excluded before object search; otherwise its green
#      border would also look like a rectangle-like non-bg structure.
#    - The old same-size output invariant is gone; output size now depends on
#      the kept rectangle crops.
#    - A new explicit assumption is added: candidate objects are black
#      rectangular frame components whose perimeter matches their bbox exactly.
# d) Placement legality:
#    The generator places each object by reserving its whole bbox, not only its
#    visible non-bg cells. That prevents nested frames, overlap, out-of-bounds
#    placements, or ambiguous touching between frames. A 1-cell border is kept
#    around the whole scene so `_bg_boundary` still recovers the orange bg.
# e) Visual observability:
#    Humans can see the grouping cue directly: the rectangles are literal black
#    containers. The important contrast with the source task is that disconnected
#    marks inside one container are still judged together, and the output is a
#    strip of cropped containers rather than a cleaned original canvas.
# f) Visual cleanliness:
#    Objects are separated by at least one bg cell in the input, and output
#    crops are separated by exactly one bg column. Each crop keeps its own frame,
#    so individual retained units remain easy to parse.
# g) Change inferability:
#    No color remapping is introduced. Colors keep their original identities;
#    only the grouping rule changes from "free connected blob" to "contents of a
#    visible rectangle", and the D14 part makes that grouping visible again in
#    the output because full frames are emitted as separate crops.
# New solver assumptions checked against the generator:
#    - Every non-legend object has a unique black rectangular frame.
#    - Every kept box contains at least one interior color mark, so crops are
#      not empty frames.
#    - At least one interior mark per box is disconnected from the frame in the
#      free-component sense, so the D9 change does not degenerate away.
#    - Surviving boxes are ordered by bbox top-left position, which is stable
#      because frames never overlap and never tie on the same top-left cell.

# --------------- solver ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
BOX_FRAME = 0
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3
OUTPUT_GAP = 1


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the smallest top-left square whose bottom edge and right edge share one
    border color, then read the legend colors from its interior.
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


def _bbox_from_pts(pts: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_bboxes(g: np.ndarray, bg: int, legend_sz: int) -> list[tuple[int, int, int, int]]:
    """Find black rectangular frame objects outside the legend."""
    frame_mask = (g == BOX_FRAME)
    frame_mask[:legend_sz, :legend_sz] = False

    boxes: list[tuple[int, int, int, int]] = []
    for pts in components_mask(frame_mask, connectivity=4):
        r0, c0, r1, c1 = _bbox_from_pts(pts)
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = box.shape
        if h < 3 or w < 3:
            continue
        if not np.all(box[0, :] == BOX_FRAME):
            continue
        if not np.all(box[-1, :] == BOX_FRAME):
            continue
        if not np.all(box[:, 0] == BOX_FRAME):
            continue
        if not np.all(box[:, -1] == BOX_FRAME):
            continue
        perimeter = 2 * h + 2 * w - 4
        if int(len(pts)) != perimeter:
            continue
        colors_inside = set(map(int, np.unique(box)))
        colors_inside.discard(bg)
        colors_inside.discard(BOX_FRAME)
        if not colors_inside:
            continue
        boxes.append((r0, c0, r1, c1))

    assert boxes, "no framed rectangular objects found"
    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes


def _collect_kept_patches(g: np.ndarray) -> tuple[int, list[np.ndarray]]:
    """Return bg and surviving rectangle crops in reading order."""
    bg = _bg_boundary(g)
    legend_sz, _legend_border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    patches: list[np.ndarray] = []
    for r0, c0, r1, c1 in _frame_bboxes(g, bg, legend_sz):
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = set(map(int, np.unique(patch)))
        colors.discard(bg)
        colors.discard(BOX_FRAME)
        if legend_colors.issubset(colors):
            continue
        patches.append(patch)

    assert patches, "no surviving framed rectangles"
    return bg, patches


def _assemble_patches(patches: list[np.ndarray], bg: int) -> np.ndarray:
    """Concatenate crops left-to-right with a single background gap."""
    assert patches, "no patches to assemble"
    height = max(int(p.shape[0]) for p in patches)
    width = sum(int(p.shape[1]) for p in patches) + OUTPUT_GAP * (len(patches) - 1)
    out = np.full((height, width), int(bg), dtype=int)

    c = 0
    for patch in patches:
        paste(out, patch, 0, c)
        c += int(patch.shape[1]) + OUTPUT_GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the rectangle-framed regions that fail the legend color test, then output those regions as separate crops.

    Input structure:
    - The background fills most of the 16x16 grid.
    - A top-left legend square has one color on its bottom edge and right edge; its interior lists the target colors.
    - Elsewhere, each candidate object is a black rectangular frame containing one or more colored marks, and those marks do not need to be mutually connected.

    Rule:
    - Read the target color set from the legend interior.
    - Ignore the legend, then detect each black rectangular frame and treat its entire bounding box as one analysis unit.
    - If the union of non-background, non-frame colors inside that box contains all target colors, discard the whole box; otherwise keep the whole box.
    - Crop every kept box to its own rectangle and concatenate the crops left-to-right with a 1-cell background gap.

    Key insight:
    The grouping cue is the visible rectangle boundary, not free 4-connectivity of the colored marks inside it. The legend still defines which color set is forbidden, but the answer is a new strip of surviving framed regions rather than a cleaned copy of the input canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg, patches = _collect_kept_patches(g)
    return _assemble_patches(patches, bg)


# --------------- generator ---------------

# -- Fixed legal slots for four boxes --
NUM_KEPT = 3
SLOT_SMALL = (5, 1, 4, 4)   # kept, no floating core required
SLOT_MED = (10, 1, 5, 5)    # kept with one floating mark
SLOT_BIG_TOP = (1, 8, 6, 6)
SLOT_BIG_BOT = (9, 8, 6, 6)

# -- Output-shape constraints --
MAX_OUTPUT_W = 23
MAX_OUTPUT_H = 7

# -- Generation retry budget --
MAX_RETRIES = 600


def _place_legend(inp: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend at top-left and place the target colors."""
    s = LEGEND_SIZE
    inp[:s, :s] = BG
    inp[s - 1, :s] = BORDER
    inp[:s, s - 1] = BORDER

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


def _fit_shape(core_h: int, core_w: int, need: int) -> list[tuple[int, int]]:
    """Choose a connected cluster shape that fits inside the safe core."""
    shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
    ]
    candidates = []
    for shape in shapes:
        if len(shape) < need:
            continue
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        if max_r < core_h and max_c < core_w:
            candidates.append(shape)
    assert candidates, "no floating-cluster shape fits inside core"
    return random.choice(candidates)


def _box_patch(
    h: int,
    w: int,
    targets: list[int],
    deleted: bool,
    target_count: int,
    *,
    floating_required: bool,
) -> np.ndarray:
    """Build one framed rectangle with interior marks."""
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = BOX_FRAME
    patch[-1, :] = BOX_FRAME
    patch[:, 0] = BOX_FRAME
    patch[:, -1] = BOX_FRAME

    required_targets = random.sample(targets, target_count)
    non_targets = [v for v in PALETTE if v not in targets]
    allowed = list(PALETTE) if deleted else non_targets + required_targets

    used: set[tuple[int, int]] = set()
    if floating_required:
        core_rows = list(range(2, h - 2))
        core_cols = list(range(2, w - 2))
        assert core_rows and core_cols, "box too small for floating interior marks"
        core_h = len(core_rows)
        core_w = len(core_cols)

        shape = _fit_shape(core_h, core_w, max(1, len(required_targets)))
        max_sr = core_h - max(r for r, _ in shape) - 1
        max_sc = core_w - max(c for _, c in shape) - 1
        start_r = random.randint(0, max_sr)
        start_c = random.randint(0, max_sc)
        cluster_cells = [(core_rows[start_r + r], core_cols[start_c + c]) for r, c in shape]

        colors = list(required_targets)
        while len(colors) < len(cluster_cells):
            colors.append(random.choice(non_targets))
        random.shuffle(colors)
        for (r, c), color in zip(cluster_cells, colors):
            patch[r, c] = color
        used.update(cluster_cells)

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if (r, c) not in used]
    random.shuffle(interior)
    min_extra = 1 if not used else 0
    extra_count = random.randint(min_extra, min(len(interior), 3))
    extra_colors = list(required_targets if not used else [])
    while len(extra_colors) < extra_count:
        extra_colors.append(random.choice(allowed))
    random.shuffle(extra_colors)
    for (r, c), color in zip(interior[:extra_count], extra_colors):
        patch[r, c] = color

    colors_inside = set(map(int, np.unique(patch)))
    colors_inside.discard(BG)
    colors_inside.discard(BOX_FRAME)
    if deleted:
        assert set(targets).issubset(colors_inside)
    else:
        assert not set(targets).issubset(colors_inside)
    return patch


def _collect_free_kept_patches(g: np.ndarray) -> tuple[int, list[np.ndarray]]:
    """Reference D14-only behavior: free connected components on all non-bg cells."""
    bg = _bg_boundary(g)
    legend_sz, _legend_border, legend_colors = _find_legend(g, bg)

    fg_mask = (g != bg)
    fg_mask[:legend_sz, :legend_sz] = False

    kept: list[tuple[tuple[int, int], np.ndarray]] = []
    for pts in components_mask(fg_mask, connectivity=4):
        colors = {int(g[r, c]) for r, c in pts}
        colors.discard(BOX_FRAME)
        if legend_colors.issubset(colors):
            continue
        r0, c0, r1, c1 = _bbox_from_pts(pts)
        kept.append(((r0, c0), g[r0 : r1 + 1, c0 : c1 + 1].copy()))

    kept.sort(key=lambda item: item[0])
    return bg, [patch for _, patch in kept]


def _degeneracy_gap(g: np.ndarray) -> int:
    """How many more kept crops free-component grouping would emit than rectangle grouping."""
    _bg, rect_patches = _collect_kept_patches(g)
    _bg2, free_patches = _collect_free_kept_patches(g)
    return len(free_patches) - len(rect_patches)


def _expected_output(inp: np.ndarray) -> np.ndarray:
    """Build the expected combo output from the placed input."""
    bg, patches = _collect_kept_patches(inp)
    return _assemble_patches(patches, bg)


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, targets)

    big_slots = [SLOT_BIG_TOP, SLOT_BIG_BOT]
    random.shuffle(big_slots)
    kept_big_h, kept_big_w = random.choice([(6, 6), (6, 5), (5, 6)])
    placements = [
        (SLOT_SMALL, False, 0, False),
        (SLOT_MED, False, 1, True),
        ((big_slots[0][0], big_slots[0][1], kept_big_h, kept_big_w), False, 2, True),
        (big_slots[1], True, NUM_TARGETS, True),
    ]

    for (r, c, h, w), deleted, target_count, floating_required in placements:
        patch = _box_patch(
            h,
            w,
            targets,
            deleted,
            target_count,
            floating_required=floating_required,
        )
        paste(inp, patch, r, c, transparent=BG)

    out = _expected_output(inp)
    if out.shape[0] > MAX_OUTPUT_H or out.shape[1] > MAX_OUTPUT_W:
        return None

    gap = _degeneracy_gap(inp)
    if gap <= 0:
        return None

    bg, free_patches = _collect_free_kept_patches(inp)
    free_out = _assemble_patches(free_patches, bg)
    if np.array_equal(free_out, out):
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
