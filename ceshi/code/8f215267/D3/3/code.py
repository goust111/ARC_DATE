# Variant of 8f215267
# Dimension: D3 — 目标对象：框架结构 (dst)
# Change: 5x11的空心矩形框架，仅由周边一圈像素组成，垂直排列在网格左半部分，间隔2行... → 框架带有内部结构
#
# Design analysis
# ===============
# Baseline understanding:
# - Original 8f215267: left-side colored frames are plain hollow 5x11 rectangles.
#   Right-side monochrome sprites are counted by color, then the count is rendered
#   as same-color dots on the frame's middle row.
# - Source task 5dbc8537 contributes the useful complexity pattern: first identify a
#   destination object that contains internal cavities / partitions, then place
#   content into the correct sub-region rather than into the whole bbox.
#
# a) Concrete variant behavior and trace:
# - Every frame keeps the same 5x11 outer rectangle and same vertical stacking layout.
# - The only structural change is inside the frame:
#   * `horizontal` frame: a same-color horizontal divider occupies the whole middle
#     interior row, splitting the inside into top and bottom chambers.
#   * `vertical` frame: a same-color vertical divider occupies the whole middle
#     interior column, splitting the inside into left and right chambers.
# - The counting rule stays the same: sprites on the right are matched to frames by
#   exact color, then removed in the output.
# - The display rule changes:
#   * horizontal-divider frame: draw dots only in the bottom chamber, using the row
#     from right to left at columns 8, 6, 4, 2 (relative to the 11-wide frame).
#   * vertical-divider frame: draw dots only in the right chamber, using the four
#     fixed slots [(3,9), (3,7), (1,9), (1,7)] relative to the frame, i.e. right
#     chamber corners ordered bottom-right, bottom-left, top-right, top-left.
# - Small example trace:
#   * Old logic: a blue frame with 3 blue sprites would place dots on the global
#     frame middle row at c1-2, c1-4, c1-6, regardless of internal structure.
#   * New logic, horizontal divider: the same 3 blue sprites place dots only in the
#     bottom chamber row, so the divider row remains intact and the dots appear below it.
#   * New logic, vertical divider: the same 3 blue sprites place dots only in the
#     right chamber slots, so no dots appear in the left chamber at all.
#   * Difference: the solver must now parse the frame's interior structure before
#     it knows where the tally should be rendered.
#
# b) Complexity calibration:
# - Only two divider types are allowed. This keeps the branch set small enough for
#   ARC-style induction while still making the visual change obvious.
# - Both orientations are forced to appear in every generated grid, and both have at
#   least one nonzero-count example, so the new rule is visibly exercised.
# - The count range remains 0..4. This preserves the original "tally" difficulty and
#   avoids introducing extra degrees of freedom unrelated to D3.
#
# c) Old implicit assumptions that break:
# - Original frame detection only relied on "large sparse monochrome component".
#   That still mostly holds, but now the decisive feature is not just hollowness:
#   the solver must recognize a full internal divider row or column.
# - Original drawing code assumes `r_mid` is the only meaningful display row.
#   That fails for horizontal-divider frames because the middle row is now occupied
#   by the divider, and it fails for vertical-divider frames because the count must
#   be confined to the right chamber.
# - Original unconditional placement on the whole frame center row would overwrite
#   the intended chamber structure and visually collapse the variant back toward the
#   seed task. The new code therefore derives slots from frame orientation.
#
# d) Placement legality:
# - Divider pixels are part of the frame component because they touch the outer border
#   by 4-connectivity; this preserves single-component frame detection.
# - Sprites are still placed in the right-side canvas only, so they cannot overlap
#   frames. `random_free_location` with padding keeps sprites mutually separated.
# - The output-first generator constructs the full desired output, removes only the
#   tally dots to obtain the input, scatters sprites, and validates with `main(inp)`.
#
# e) Visual observability:
# - Internal dividers are thick, same-color lines; they are easy to see in the input.
# - Output dots clearly appear in a chamber rather than across the full center row.
# - Since both divider orientations and both display branches are present in each
#   sample, a human can infer "frame interior structure controls the tally chamber"
#   from a few examples.
#
# f) Visual cleanliness:
# - The left region still contains only stacked frames.
# - The right region still contains only small sprites.
# - Dots are restricted to sparse fixed slots, so the chambers remain readable instead
#   of becoming solid filled blocks.
#
# g) Color-rule audit:
# - No new color mapping is introduced. Colors retain the original meaning:
#   frame color == sprite color == tally color.
# - The only new latent variable is frame orientation, and it is fully visible from
#   the geometry of the frame itself.
#
# New main() assumptions and how the generator guarantees them:
# - Assumption: true frames are exactly 5x11 and contain exactly one centered full
#   divider row or divider column. The generator constructs only such frames.
# - Assumption: non-frame components are smaller sprites. The generator bounds sprite
#   size to <= 4x3, so no sprite can mimic a 5x11 partitioned frame.
# - Assumption: at most 4 dots are needed for any frame. The generator caps counts at 4.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 6)        # 3-5 frames
N_DISTRACTORS_RANGE = (1, 3)   # 1-2 distractor sprites
TOTAL_PATTERNS_RANGE = (5, 11) # 5-10 total sprites
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300

HORIZONTAL = "horizontal"
VERTICAL = "vertical"


def _frame_orientation(patch: np.ndarray, color: int) -> str | None:
    """Return the frame's divider orientation if this is a valid partitioned frame."""
    if patch.shape != (FRAME_H, FRAME_W):
        return None

    # Outer border must be solid.
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return None

    horizontal = bool(np.all(patch[FRAME_H // 2, 1:-1] == color))
    vertical = bool(np.all(patch[1:-1, FRAME_W // 2] == color))
    if horizontal == vertical:
        return None
    return HORIZONTAL if horizontal else VERTICAL


def _dot_slots(orientation: str) -> list[tuple[int, int]]:
    """Relative (r,c) slots for counts 1..4 inside the designated chamber."""
    if orientation == HORIZONTAL:
        row = FRAME_H - 2
        return [(row, 8), (row, 6), (row, 4), (row, 2)]
    return [(3, 9), (3, 7), (1, 9), (1, 7)]


def _frame_patch(color: int, bg: int, orientation: str, n_dots: int) -> np.ndarray:
    """Build a 5x11 frame with a centered divider and `n_dots` rendered in its target chamber."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if orientation == HORIZONTAL:
        patch[FRAME_H // 2, 1:-1] = color
    else:
        patch[1:-1, FRAME_W // 2] = color
    for rr, cc in _dot_slots(orientation)[: int(n_dots)]:
        patch[int(rr), int(cc)] = color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per partitioned frame and render the tally in the frame's target chamber.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = tuple(map(int, comp["bbox"]))
        h = bbox_h(b)
        w = bbox_w(b)
        r0, c0, r1, c1 = b
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        orientation = _frame_orientation(patch, int(comp["color"]))
        if h == FRAME_H and w == FRAME_W and orientation is not None:
            frames.append({"comp": comp, "orientation": orientation})
        else:
            patterns.append(comp)

    # Remove sprites from the output.
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    # For each frame, count matching sprites and draw dots in the chamber implied by the divider.
    for frame in frames:
        comp = frame["comp"]
        col = int(comp["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        for rr, cc in _dot_slots(str(frame["orientation"]))[: int(num_patterns)]:
            out[int(r0 + rr), int(c0 + cc)] = col

    return out


def _sample_orientations(rng: np.random.Generator, n_frames: int) -> list[str]:
    """Force every sample to contain both branch types."""
    orientations = [HORIZONTAL, VERTICAL]
    for _ in range(n_frames - 2):
        orientations.append(HORIZONTAL if int(rng.integers(0, 2)) == 0 else VERTICAL)
    rng.shuffle(orientations)
    return orientations


def _sample_counts(
    rng: np.random.Generator, orientations: list[str], total_match: int
) -> list[int]:
    """Distribute matched sprites across frames, with both orientations visibly exercised."""
    counts = [0] * len(orientations)

    horizontal_ids = [i for i, ori in enumerate(orientations) if ori == HORIZONTAL]
    vertical_ids = [i for i, ori in enumerate(orientations) if ori == VERTICAL]
    counts[int(rng.choice(horizontal_ids))] += 1
    counts[int(rng.choice(vertical_ids))] += 1
    remaining = int(total_match - 2)

    order = [int(x) for x in rng.permutation(len(orientations))]
    for pos, idx in enumerate(order):
        cap = MAX_DOTS_PER_FRAME - counts[idx]
        future_cap = sum(MAX_DOTS_PER_FRAME - counts[j] for j in order[pos + 1 :])
        lo = max(0, remaining - future_cap)
        hi = min(cap, remaining)
        add = int(rng.integers(lo, hi + 1))
        counts[idx] += add
        remaining -= add

    if remaining != 0:
        raise ValueError("failed to distribute pattern counts")
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino of the given area inside an h x w bounding box."""
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.append(nb)
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    """Generate a small random connected sprite, tight-cropped to its bounding box."""
    area = int(rng.integers(*SPRITE_AREA_RANGE))
    dims = [
        (h, w)
        for h in range(1, SPRITE_MAX_H + 1)
        for w in range(1, SPRITE_MAX_W + 1)
        if h * w >= area
    ]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = int(total_patterns - n_distractors)

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    orientations = _sample_orientations(rng, n_frames)
    counts = _sample_counts(rng, orientations, total_match)

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    # --- build output (partitioned frames with dots) ---
    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, orientation, count) in enumerate(zip(frame_colors, orientations, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, orientation, count), r0, FRAME_C0)

    # --- derive input (dots removed, sprites scattered right) ---
    inp = out.copy()
    for r0, orientation, count in zip(frame_rows, orientations, counts):
        for rr, cc in _dot_slots(orientation)[: int(count)]:
            inp[int(r0 + rr), int(FRAME_C0 + cc)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
