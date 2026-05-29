# Combo variant of 8f215267 (index 25)
# combo_spec: D4/1+D5/0+D11/0
# Name: 计数取反+左侧放置+统一颜色

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for combo D4/1 + D5/0 + D11/0
#
# source_task:
#   - Base task 8f215267: stack hollow colored frames on the left, scatter
#     small monochrome sprites on the right, then convert per-frame sprite
#     counts into center-row dots.
#   - Reference spec keeps the physical layout stable: 5x11 hollow frames,
#     23 columns total, right-side sprite region, and at most 4 legal dot slots.
#
# a) concretization and behavior trace:
#   - Keep the same visible objects as the original task:
#       left = colored hollow frames,
#       right = small scattered sprites,
#       input frame interiors empty.
#   - Suppose three frames have true matching sprite counts [4, 1, 0].
#     Original task:
#       - draw [4, 1, 0] dots,
#       - place them from the right inner wall,
#       - color each dot with that frame's color.
#     Combo task:
#       - draw [0, 3, 4] dots because display = 4 - N,
#       - place them from the left inner wall,
#       - color every displayed dot with the same fixed marker color.
#     So the same input now changes in three coupled ways at once:
#       quantity is inverted, direction is mirrored, and color is detached
#       from the frame identity.
#
# b) complexity calibration:
#   - The matching relation itself stays simple and visible: sprite color still
#     maps to frame color one-to-one, so the user only has to discover how the
#     tally is displayed, not how objects are paired.
#   - The dot vocabulary remains capped at 0..4 because the fixed 5x11 frame
#     has exactly four readable slots on the center row.
#   - To keep the combo visible in only 2-3 examples, the generator forces:
#       1. one frame with N = 0,
#       2. one frame with N = 4,
#       3. one frame with 1 <= N <= 3,
#       4. at least one displayed tally of 2 or 3 dots,
#       5. at least two frames with positive displayed tallies.
#     These jointly expose inversion, left anchoring, and cross-frame color
#     uniformity without adding extra object types or layout freedom.
#
# c) hidden-assumption audit for main():
#   - bg = get_bg(g):
#       still valid because frames are hollow and sprites remain sparse.
#   - find_components(... monochromatic=True, connectivity=4):
#       still valid because every frame and every sprite is a single-color
#       component in the input. The fixed marker color only appears in output.
#   - is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7):
#       still valid because input frames remain empty hollow rectangles.
#   - Removing all non-frame components by writing bg:
#       still safe because generator keeps every sprite in the right region,
#       disjoint from frames.
#   - Counting by exact color equality between frame and sprite:
#       still valid because frame colors remain distinct and distractor colors
#       never match a frame color.
#   - New display rule:
#       the solver now assumes each frame has four legal left-anchored slots
#       and that matching counts never exceed slot capacity.
#       The generator guarantees 0 <= N <= 4 for every frame.
#   - New color rule:
#       the solver writes a fixed marker color that is independent of frame
#       color. The generator reserves this marker color so it is distinct from
#       background, frame borders, and distractor sprites.
#
# d) placement legality:
#   - Frames still occupy the left block and are vertically stacked with gaps.
#   - Sprites are still pasted only into the separate right canvas and then
#     overlaid at column 13, so they never overlap a frame.
#   - random_free_location(..., padding=1, padding_connectivity=8) keeps
#     sprites mutually separated and in-bounds.
#   - Max displayed dots is fixed at 4, which fits the local columns 2,4,6,8
#     inside a width-11 frame without touching corners or borders.
#
# e) visual observability:
#   - Inversion is visible because samples include empty, saturated, and
#     intermediate true counts, so users can see "more sprites -> fewer dots".
#   - Left placement is visible because at least one frame displays 2 or 3
#     dots, which makes the left-to-right fill direction obvious.
#   - Uniform color is visible because at least two different frame colors show
#     positive tallies using the same marker color in the same sample.
#   - No hidden numeric ordering of color ids is used.
#
# f) cleanliness:
#   - Left region contains only frames plus their interior markers in output.
#   - Right region contains only sprites in input and becomes empty in output.
#   - Dots stay on one row with one-cell gaps, so the marker channel is clean
#     even after detaching dot color from frame color.
#
# g) color-change inferability:
#   - A single fixed marker color is easier to infer than a many-to-many color
#     remapping. The dot color is globally constant across all frames and all
#     samples, while border colors continue to indicate sprite ownership.
#   - Reserving the marker color away from every input object prevents false
#     cues such as distractors sharing the output marker color.
#
# h) quantitative non-degeneration target:
#   - Expressive samples satisfy all of:
#       1. counts contain 0, 4, and at least one intermediate value,
#       2. displayed counts contain at least one value in {2, 3},
#       3. at least two frames have displayed dots.
#   - These are the witness conditions for the three combo dimensions:
#       inversion witness, left-anchor witness, and multi-frame uniform-color
#       witness. The generator samples only from this expressive subset.
#
# i) new assumptions introduced by the modified main():
#   - The intended marker color is globally fixed rather than frame-dependent.
#   - The displayed readout is computed from slot capacity minus sprite count.
#   - The displayed readout starts at the left inner wall.
#   - generate_input() enforces all three by constructing validated outputs
#     with the same logic and erasing only those marker cells to form inputs.
# ---------------------------------------------------------------------------


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
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- palette ---
DOT_COLOR = 2
AVAILABLE_COLORS = [c for c in range(10) if c != DOT_COLOR]

# --- retry budget ---
MAX_ATTEMPTS = 300
COUNT_SAMPLE_ATTEMPTS = 300


def _slot_columns_from_left(c0: int, c1: int) -> list[int]:
    """Return legal center-row slot columns from the left inner wall inward."""
    return [int(cc) for cc in range(int(c0) + 2, int(c1), 2)]


def _local_dot_cols_from_left(n_dots: int) -> list[int]:
    """Return local patch columns for the first n left-anchored marker slots."""
    return _slot_columns_from_left(0, FRAME_W - 1)[: int(n_dots)]


def _split_components(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    """Return background color, detected frames, and detected non-frame patterns."""
    bg = int(get_bg(grid))
    regions = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

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
    return bg, frames, patterns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert same-color sprite counts into inverted left-anchored marker dots.

    Input structure:
    - The most common color is the background.
    - Several hollow rectangular frames with distinct border colors sit on the left.
    - Small monochrome sprites are scattered outside the frames on the right.
    - Frame interiors are empty in the input; only sprite colors determine which frame they belong to.

    Rule:
    - Remove every sprite from the grid.
    - For each frame of color C, count how many sprites share color C; call this N.
    - The frame has four legal center-row marker slots, so draw (4 - N) dots rather than N.
    - Fill those dots from the left inner wall toward the right, every 2 columns.
    - Use one fixed marker color for all frames instead of reusing each frame's border color.

    Key insight:
    - Frame colors still define the counting groups, but the displayed readout is a transformed code:
      the quantity is inverted, the placement direction is mirrored, and the marker color is globally shared.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg, frames, patterns = _split_components(g)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        frame_color = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == frame_color)

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        slots = _slot_columns_from_left(c0, c1)
        dots_to_draw = max(0, len(slots) - num_patterns)
        for cc in slots[:dots_to_draw]:
            out[int(r_mid), int(cc)] = DOT_COLOR

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a hollow frame whose interior markers use the global fixed dot color."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    patch[mid_r, _local_dot_cols_from_left(n_dots)] = DOT_COLOR
    return patch


def _sample_match_counts(rng: np.random.Generator, n_frames: int, n_distractors: int) -> list[int] | None:
    """
    Sample true same-color sprite counts per frame from the expressive subset.

    Witness constraints:
    - include 0, 4, and one intermediate count for inversion,
    - include one displayed tally of 2 or 3 for visible left anchoring,
    - include at least two positive displayed tallies for visible uniform color.
    """
    lo_total, hi_total = TOTAL_PATTERNS_RANGE
    for _ in range(COUNT_SAMPLE_ATTEMPTS):
        counts = [int(rng.integers(0, MAX_DOTS_PER_FRAME + 1)) for _ in range(n_frames)]
        if 0 not in counts:
            continue
        if MAX_DOTS_PER_FRAME not in counts:
            continue
        if not any(0 < c < MAX_DOTS_PER_FRAME for c in counts):
            continue

        display_counts = [MAX_DOTS_PER_FRAME - c for c in counts]
        if not any(v in (2, 3) for v in display_counts):
            continue
        if sum(v > 0 for v in display_counts) < 2:
            continue

        total_patterns = sum(counts) + n_distractors
        if lo_total <= total_patterns < hi_total:
            return counts
    return None


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
    match_counts = _sample_match_counts(rng, n_frames, n_distractors)
    if match_counts is None:
        return None

    palette = rng.choice(np.array(AVAILABLE_COLORS, dtype=int), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    display_counts = [MAX_DOTS_PER_FRAME - n for n in match_counts]
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, n_dots) in enumerate(zip(frame_colors, display_counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, n_dots), r0, FRAME_C0)

    inp = out.copy()
    for r0, n_dots in zip(frame_rows, display_counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + cc for cc in _local_dot_cols_from_left(n_dots)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, match_counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, int(color), bg)
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
    """
    Return validated inputs for the inverted-left-fixed-color combo.

    Guarantees:
    - every frame gets 0..4 matching sprites,
    - every sample contains inversion witnesses and a visible left-fill witness,
    - at least two positive frame tallies share the same fixed dot color,
    - the fixed dot color is absent from the input palette,
    - only inputs whose output exactly matches main() are returned.
    """
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
