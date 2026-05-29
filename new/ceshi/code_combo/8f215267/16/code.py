# Combo variant of 8f215267 (index 16)
# combo_spec: D5/2+D7/0
# Name: 保留精灵+对角线点

from __future__ import annotations

"""
Design analysis for the combo variant D5/2 + D7/0

a) Concrete behavior and trace
- The source task has vertically stacked hollow color frames on the left and scattered
  monochrome sprites on the right. The original output deletes all sprites, then writes
  a same-color tally on each frame's center row.
- This combo changes two stages together:
  1. D7/0: the sprites are not erased in the output.
  2. D5/2: the tally no longer occupies one horizontal row; it is written on the interior
     diagonal from the lower-right corner toward the upper-left.
- Example with one frame color C and three matching sprites:
  source task output = right-side sprites disappear, and frame C gets 3 marks at
  `(r_mid, c1-2)`, `(r_mid, c1-4)`, `(r_mid, c1-6)`;
  combo output = the same right-side sprites remain visible, and frame C instead gets
  3 marks at `(r1-1, c1-1)`, `(r1-2, c1-2)`, `(r1-3, c1-3)`.
- The solver therefore has to infer both "what stays unchanged" and "where the new marks
  are added".

b) Complexity calibration
- A 5x11 hollow frame has a 3x9 interior, so the chosen diagonal can hold only 3 cells.
  Keeping the old cap of 4 would create impossible displays, so the generator lowers the
  maximum matching count per frame to 3.
- To keep the diagonal visually evident, generation insists that at least one frame has
  count >= 2 and that at least two frames are active. Otherwise the task could degenerate
  into a single-frame tally that gives too little within-example evidence for the fused rule.
- The overall scene structure stays close to the source task: 3-4 frames, 5-10 total
  sprites, and 1-2 distractor colors. That keeps the rule readable without introducing
  extra object types.

c) Implicit-assumption audit of `main()`
- `out = g.copy()` still works, but its meaning changes: we now preserve every source
  sprite and only add new dots inside frames.
- Component classification still assumes frames are the large sparse same-color components
  and sprites are the smaller compact ones. The generator preserves that separation.
- The source task's sprite-erasing loop depended on "every non-frame component vanishes".
  That assumption is invalid here and must be removed.
- The original horizontal drawing code assumed a single display row and width-only
  capacity. The combo breaks both assumptions. The replacement must compute full `(r, c)`
  pairs on a `(-1, -1)` diagonal clipped to the frame interior.
- New assumption introduced by the diagonal rule: frame interiors must be tall enough and
  empty before drawing. The fixed 5x11 frame shape gives 3 interior rows, and the generator
  constructs inputs with no pre-existing frame-interior marks.

d) Placement legality
- Frames stay in the left block with fixed width and vertical gaps, so they never overlap.
- Sprites are still generated on a separate right-side canvas and pasted at column 13,
  leaving at least one background column between frames and sprites.
- `random_free_location(..., padding=1, padding_connectivity=8)` keeps sprites disjoint and
  in bounds.
- The diagonal dots are only added at strict interior cells, so they never touch outside
  regions or exceed frame bounds.

e) Visual observability
- A human comparing examples can see that the right-side sprites remain unchanged from input
  to output, so the transformation is additive rather than absorptive.
- The new marks line up from lower-right to upper-left inside the frame, which contrasts
  with the source task's middle-row layout.
- Requiring at least one frame with count >= 2 gives repeated evidence for the diagonal
  rather than relying on a single ambiguous added pixel.

f) Visual cleanliness
- The left area still contains only frames and their added dots; the right area still
  contains only sprites. This keeps regions semantically separated.
- Because the output is formed by overlaying dots onto the unchanged input, there is no
  destructive overwrite of sprites and no confusing multi-object collisions.

g) Color inference
- Colors retain a one-to-one "same color means same bucket" meaning.
- Distractor sprite colors do not match any frame color, so they remain visible negative
  evidence: they stay in place but contribute no diagonal dots.

Generator guarantees for the new solver assumptions
- Matching counts never exceed `MAX_DOTS_PER_FRAME = 3`.
- At least one frame has count >= 2, making the diagonal layout observable.
- At least two frames receive nonzero counts, so color-bucket evidence is visible inside
  each single example instead of relying only on cross-example variation.
- Every generated sample is validated with `np.array_equal(main(inp), out)`.
"""

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


def _diagonal_dot_positions_from_bbox(
    bbox: tuple[int, int, int, int], n_dots: int
) -> list[tuple[int, int]]:
    """Return strict interior cells from lower-right toward upper-left."""
    r0, c0, r1, c1 = map(int, bbox)
    cells: list[tuple[int, int]] = []
    for i in range(int(n_dots)):
        rr = int(r1 - 1 - i)
        cc = int(c1 - 1 - i)
        if rr > int(r0) and cc > int(c0):
            cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites for each frame and display the tally as diagonal dots.

    Input structure:
    - The background is the most common color.
    - Several hollow rectangular frames of distinct colors are stacked on the left.
    - Small monochrome sprites of frame colors and distractor colors are scattered on the right.

    Rule:
    - Keep every sprite exactly where it is.
    - For each frame of color C, count how many small sprites also have color C.
    - Add that many dots inside the frame along the diagonal running from the
      lower-right interior corner toward the upper-left, one step per dot.

    Key insight:
    - The right-side sprites are not consumed; they remain as visible evidence.
      Each frame acts as a same-color counter whose tally is shown by newly added
      diagonal marks inside the hollow box.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        for rr, cc in _diagonal_dot_positions_from_bbox(f["bbox"], num_patterns):
            out[int(rr), int(cc)] = col

    return out


# --------------- generator (combo variant) ---------------

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
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 3
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_diagonal_cells(n_dots: int) -> list[tuple[int, int]]:
    """Local diagonal cells inside a FRAME_H x FRAME_W frame patch."""
    return _diagonal_dot_positions_from_bbox((0, 0, FRAME_H - 1, FRAME_W - 1), n_dots)


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a hollow frame patch with optional diagonal tally dots."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for rr, cc in _frame_diagonal_cells(n_dots):
        patch[int(rr), int(cc)] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """Distribute matches across frames, respecting the diagonal capacity."""
    if total_match > n_frames * MAX_DOTS_PER_FRAME:
        raise ValueError("total_match exceeds per-frame diagonal capacity")

    counts = [0] * n_frames
    for _ in range(128):
        counts = [0] * n_frames
        order = rng.permutation(n_frames)
        remaining = total_match
        for pos, idx in enumerate(order):
            frames_left = n_frames - pos - 1
            lo = max(0, remaining - MAX_DOTS_PER_FRAME * frames_left)
            hi = min(MAX_DOTS_PER_FRAME, remaining)
            val = int(rng.integers(lo, hi + 1))
            counts[int(idx)] = val
            remaining -= val
        if max(counts) >= 2 and sum(1 for x in counts if x > 0) >= 2:
            return counts
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino of the given area inside an h x w box."""
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
    """Generate one small connected monochrome sprite."""
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
    total_match = total_patterns - n_distractors
    if total_match <= 0:
        return None

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, n_frames, total_match)
    if (
        sum(counts) != total_match
        or max(counts) < 2
        or sum(1 for x in counts if x > 0) < 2
    ):
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    # Input keeps empty frames on the left and all source sprites on the right.
    inp = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, color in enumerate(frame_colors):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(inp, _frame_patch(color, bg, 0), r0, FRAME_C0)

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

    # Output keeps the sprites and adds diagonal tally dots to the matching frames.
    out = inp.copy()
    for r0, color, count in zip(frame_rows, frame_colors, counts):
        for rr, cc in _frame_diagonal_cells(count):
            out[int(r0 + rr), int(FRAME_C0 + cc)] = int(color)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
