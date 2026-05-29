# Combo variant of 8f215267 (index 22)
# combo_spec: D8/1+D9/2
# Name: 每框架独立输出+网格排列

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# source_task: 8f215267
# combo_spec: D8/1 + D9/2
#
# a) Concrete behavior and trace
# - Original behavior: a tall 23-column grid contains 3-4 large frames in one left-side column.
#   Sprites live in a separate right strip. Output keeps the whole canvas and only adds dots back
#   into the matching frames.
# - This combo changes two linked stages at once.
#   1. Input layout becomes a 2x2 frame arrangement, so color-matching must work across both rows
#      and both columns rather than along one vertical stack.
#   2. Output no longer preserves the full input canvas. Instead, each solved frame is cropped to
#      its own 5x7 patch and the four patches are repacked into a compact 2x2 mosaic in row-major
#      order (top-left, top-right, bottom-left, bottom-right).
# - Example trace:
#   Input: four frames at positions TL/TR/BL/BR with colors [red, blue, yellow, green]. Suppose the
#   scattered sprites contain [red, red, blue, green, distractor]. The solver counts by color, so
#   the solved patches become [red with 2 dots, blue with 1 dot, yellow with 0 dots, green with
#   1 dot]. Output is an 11x15 compact grid made only from those four cropped frame patches.
#
# b) Complexity calibration
# - Fixed 2x2 layout keeps the fusion visually obvious without introducing too many cases.
# - Frames are reduced to 5x7 so that a 2x2 input layout still leaves a generous corridor region
#   for sprite placement. This also lowers dot capacity to 2, which is a necessary geometric
#   consequence rather than a new logical rule.
# - At least three frames are active in generated samples, so the reader sees repeated evidence of
#   both the counting rule and the frame-separation rule.
#
# c) Implicit-assumption review of main()
# - The original area heuristic for detecting frames becomes fragile after shrinking frames and
#   moving sprites into the same overall bounding box. The solver therefore switches to an exact
#   hollow-rectangle test.
# - The old solver assumed the final output had the same shape as the input. That no longer holds.
#   After counting, the solver must sort frames by spatial order and synthesize a new compact grid.
# - The dot rule still assumes dots sit on the center row and step left by 2. With width 7 this
#   safely encodes counts 0-2.
#
# d) Placement legality
# - Input generation is still output-first: first choose four solved frame patches, then derive the
#   larger input by removing dots and scattering sprites.
# - Sprites are only allowed inside the union rectangle of the four frames but never inside any
#   frame bbox, so they remain in visible corridor background and cannot overlap frames.
# - random_free_location plus padding enforces non-overlap and spacing between sprites.
#
# e) Visual observability
# - A human can see two persistent signals across examples:
#   1. The input has four frames in a 2x2 arrangement instead of one vertical column.
#   2. The output is a smaller repacked mosaic made from four cropped frame patches rather than the
#      full input canvas.
# - The ordering cue is human-visible because it follows reading order from the input geometry, not
#   hidden numeric color ids.
#
# f) Visual cleanliness
# - Frames use distinct colors and remain simple hollow rectangles.
# - Sprites stay in corridor background, so they do not sit inside frame interiors or touch frame
#   borders.
# - The compact output uses 1-cell separators, which keeps the four cropped patches distinct.
#
# g) Change inferability
# - Color usage remains direct equality matching between sprite color and frame color, so no opaque
#   remapping needs to be inferred.
# - The new inference burden is structural rather than symbolic: detect the four input frames,
#   count same-color sprites, solve each frame, then repack the solved frame patches by row-major
#   frame order.
#
# h) Anti-degeneration signal
# - Two measurable combo signals are enforced by the generator:
#   1. Input frame columns always have cardinality 2 and input frame rows always have cardinality 2.
#   2. Output shape is always the compact 11x15 mosaic, which differs from the larger input shape.
# - If either signal disappears, the instance has drifted back toward the original task.
#
# i) New solver assumptions audited against the generator
# - main() assumes every true frame is an empty monochrome hollow rectangle when observed in input.
# - main() assumes non-frame monochrome components are sprites or distractors and can be discarded.
# - main() assumes row-major frame order is unambiguous because the four frames occupy two distinct
#   rows and two distinct columns with clear gaps.
# - generate_input() guarantees all three conditions.


FRAME_H = 5
FRAME_W = 7
DOT_CAPACITY = 2

OUT_GAP = 1
OUT_ROWS = [0, FRAME_H + OUT_GAP]
OUT_COLS = [0, FRAME_W + OUT_GAP]

TOP_PAD_CHOICES = (1, 2)
LEFT_PAD_CHOICES = (1, 2)
ROW_GAP_CHOICES = (6, 7, 8)
COL_GAP_CHOICES = (6, 7, 8)
BOTTOM_PAD_CHOICES = (1, 2, 3)
RIGHT_PAD_CHOICES = (1, 2, 3)

N_FRAMES = 4
MIN_ACTIVE_FRAMES = 3
TOTAL_MATCH_RANGE = (3, 7)   # inclusive lower, exclusive upper for rng.integers
N_DISTRACTORS_RANGE = (1, 3)

SPRITE_AREA_RANGE = (2, 7)
SPRITE_MAX_H = 3
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MAX_ATTEMPTS = 300


def _is_hollow_frame(comp: dict) -> bool:
    """Return True iff the component is an exact hollow rectangular perimeter."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    if h < 5 or w < 5:
        return False

    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    if len(coords) != (2 * (h + w) - 4):
        return False

    for rr in range(r0, r1 + 1):
        if (rr, c0) not in coords or (rr, c1) not in coords:
            return False
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in coords or (r1, cc) not in coords:
            return False
    return True


def _frame_patch(color: int, bg: int, h: int, w: int, n_dots: int) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = h // 2
    for i in range(int(n_dots)):
        cc = w - 3 - 2 * i
        if cc > 0:
            patch[mid_r, cc] = color
    return patch


def _assemble_output(frame_patches: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = 2 * FRAME_H + OUT_GAP
    out_w = 2 * FRAME_W + OUT_GAP
    out = np.full((out_h, out_w), bg, dtype=int)
    for patch, r0, c0 in zip(frame_patches, [OUT_ROWS[0], OUT_ROWS[0], OUT_ROWS[1], OUT_ROWS[1]], [OUT_COLS[0], OUT_COLS[1], OUT_COLS[0], OUT_COLS[1]]):
        paste(out, patch, r0, c0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count sprites by frame color, solve each frame locally, then repack the solved frames.

    Input structure:
    - bg is the most frequent color.
    - Four hollow rectangular frames of distinct colors are arranged as a visible 2x2 layout.
    - Small monochrome sprites are scattered only in the corridor background between those frames.

    Rule:
    - Detect the four frames and all non-frame sprite components.
    - For each frame color C, count how many scattered sprites also use C.
    - Convert each frame into its own solved patch by drawing that many dots on the center row,
      stepping left by 2 from the right interior side.
    - Crop the four solved frame patches and place them into a compact 2x2 output mosaic in the
      same row-major order as the input frames.

    Key insight: The task is no longer "edit the big canvas in place"; the big input is only a
    source layout from which four independent frame results must be extracted and repacked.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_hollow_frame(comp):
            frames.append(comp)
        else:
            patterns.append(comp)

    if not frames:
        return g.copy()

    frames.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))

    patches: list[np.ndarray] = []
    for frame in frames:
        color = int(frame["color"])
        h = bbox_h(frame["bbox"])
        w = bbox_w(frame["bbox"])
        count = sum(1 for pat in patterns if int(pat["color"]) == color)
        patches.append(_frame_patch(color, bg, h, w, count))

    return _assemble_output(patches, bg)


def _sample_counts(rng: np.random.Generator, total_match: int) -> list[int]:
    """Distribute counts across 4 frames with cap 2 and at least MIN_ACTIVE_FRAMES active."""
    counts = [0] * N_FRAMES
    remaining = total_match
    active = set(int(x) for x in rng.choice(np.arange(N_FRAMES), size=MIN_ACTIVE_FRAMES, replace=False))
    for idx in active:
        counts[idx] = 1
    remaining -= MIN_ACTIVE_FRAMES

    while remaining > 0:
        choices = [i for i in range(N_FRAMES) if counts[i] < DOT_CAPACITY]
        if not choices:
            raise ValueError("count sampling exhausted capacity")
        idx = int(choices[int(rng.integers(0, len(choices)))])
        counts[idx] += 1
        remaining -= 1

    rng.shuffle(counts)
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
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


def _input_layout(rng: np.random.Generator) -> tuple[list[int], list[int], int, int]:
    top = int(rng.choice(TOP_PAD_CHOICES))
    left = int(rng.choice(LEFT_PAD_CHOICES))
    row_gap = int(rng.choice(ROW_GAP_CHOICES))
    col_gap = int(rng.choice(COL_GAP_CHOICES))
    bottom_pad = int(rng.choice(BOTTOM_PAD_CHOICES))
    right_pad = int(rng.choice(RIGHT_PAD_CHOICES))

    frame_rows = [top, top + FRAME_H + row_gap]
    frame_cols = [left, left + FRAME_W + col_gap]
    height = frame_rows[-1] + FRAME_H + bottom_pad
    width = frame_cols[-1] + FRAME_W + right_pad
    return frame_rows, frame_cols, height, width


def _corridor_mask(shape: tuple[int, int], frame_bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0 = min(b[0] for b in frame_bboxes)
    c0 = min(b[1] for b in frame_bboxes)
    r1 = max(b[2] for b in frame_bboxes)
    c1 = max(b[3] for b in frame_bboxes)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for fr0, fc0, fr1, fc1 in frame_bboxes:
        mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    return mask


def _build_output(bg: int, frame_colors: list[int], counts: list[int]) -> np.ndarray:
    patches = [_frame_patch(color, bg, FRAME_H, FRAME_W, count) for color, count in zip(frame_colors, counts)]
    return _assemble_output(patches, bg)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    frame_rows, frame_cols, height, width = _input_layout(rng)
    total_match = int(rng.integers(*TOTAL_MATCH_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))

    palette = rng.choice(np.arange(10), size=1 + N_FRAMES + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + N_FRAMES]]
    distractor_colors = [int(x) for x in palette[1 + N_FRAMES : 1 + N_FRAMES + n_distractors]]

    try:
        counts = _sample_counts(rng, total_match)
    except ValueError:
        return None

    out = _build_output(bg, frame_colors, counts)

    inp = np.full((height, width), bg, dtype=int)
    frame_bboxes: list[tuple[int, int, int, int]] = []
    frame_positions: list[tuple[int, int]] = []
    idx = 0
    for r0 in frame_rows:
        for c0 in frame_cols:
            paste(inp, _frame_patch(frame_colors[idx], bg, FRAME_H, FRAME_W, 0), r0, c0)
            frame_bboxes.append((r0, c0, r0 + FRAME_H - 1, c0 + FRAME_W - 1))
            frame_positions.append((r0, c0))
            idx += 1

    allowed = _corridor_mask(inp.shape, frame_bboxes)
    placement_canvas = np.full(inp.shape, -1, dtype=int)
    placement_canvas[allowed] = bg

    pattern_colors = [color for color, count in zip(frame_colors, counts) for _ in range(count)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                placement_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(placement_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    sprite_mask = placement_canvas >= 0
    sprite_mask &= placement_canvas != bg
    inp[sprite_mask] = placement_canvas[sprite_mask]

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
